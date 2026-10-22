"""Среда агента-прогнозиста: агент сам пишет 24 числа, инструменты дают контекст.

Варианты (лесенка контекста, она же абляция):
  series   — только анализ ряда и опорные профили;
  calendar — + календарь и обезличенные сведения о здании;
  weather  — + температура (история и прогноз на сутки);
  full     — + новости и город.

Режим masked_dates: агент не видит настоящих дат (только «день -3, вторник»),
названия праздников скрыты. Сравнение с обычным режимом показывает, помогает ли
агенту память LLM о конкретных датах — это проверка на утечку через предобучение.
"""
import json
import numpy as np
import pandas as pd

import one_agent_forecasting.src.context as context

VARIANT_TOOLS = {
    "no_tools": ["submit_forecast"],          # LLM без инструментов: ряд прямо в сообщении (как LLMTime)
    "series":   ["describe_series", "get_values", "reference_profiles", "submit_forecast"],
    "calendar": ["describe_series", "get_values", "reference_profiles", "calendar", "building_info",
                 "submit_forecast"],
    "weather":  ["describe_series", "get_values", "reference_profiles", "calendar", "building_info",
                 "weather", "submit_forecast"],
    "full":     ["describe_series", "get_values", "reference_profiles", "calendar", "building_info",
                 "weather", "news_search", "submit_forecast"],
}

TOOL_SPECS = {
    "describe_series": {
        "description": "Summary of the consumption history: level, recent trend, typical daily profile "
                       "for weekdays and weekends, peaks, daily/weekly autocorrelation, unusual recent days.",
        "parameters": {"type": "object", "properties": {}}},
    "get_values": {
        "description": "Hourly values of the history. hours_ago_start: how many hours before the forecast start "
                       "to begin (1..336); n_hours: how many hours to return (max 72).",
        "parameters": {"type": "object", "properties": {
            "hours_ago_start": {"type": "integer"}, "n_hours": {"type": "integer"}},
            "required": ["hours_ago_start", "n_hours"]}},
    "reference_profiles": {
        "description": "Ready-made 24-hour reference profiles aligned with the forecast hours: yesterday, "
                       "same weekday last week, mean of the same weekday over the last 2 weeks, "
                       "median weekday profile, median weekend profile.",
        "parameters": {"type": "object", "properties": {}}},
    "calendar": {
        "description": "Calendar of the forecast day and the history days: weekday and public holidays "
                       "of the building's country/region.",
        "parameters": {"type": "object", "properties": {}}},
    "building_info": {
        "description": "Anonymized building metadata: use type, floor area, year built, country, climate.",
        "parameters": {"type": "object", "properties": {}}},
    "weather": {
        "description": "Outdoor temperature (°C): daily min/mean/max for the history and the hourly "
                       "day-ahead temperature forecast for the forecast day (a forecast, it has errors).",
        "parameters": {"type": "object", "properties": {}}},
    "news_search": {
        "description": "World news headlines for a given date (YYYY-MM-DD), as published before the forecast "
                       "start. Optional query keywords filter the headlines (e.g. 'power outage storm London').",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string"}, "query": {"type": "string"}}, "required": ["date"]}},
    "submit_forecast": {
        "description": "FINAL step. Submit exactly 24 non-negative hourly values (kWh) for the forecast day, "
                       "hour 00 to hour 23, and a short justification.",
        "parameters": {"type": "object", "properties": {
            "values": {"type": "array", "items": {"type": "number"}, "minItems": 24, "maxItems": 24},
            "reasoning": {"type": "string"}}, "required": ["values", "reasoning"]}},
}

SYSTEM_PROMPT = """You are an energy analyst. Forecast the hourly electricity consumption (kWh) \
of one building for the next 24 hours: {forecast_day}, hours 00-23. You have {history} hours of \
history ending right before the forecast starts.

Rules:
- Use ONLY information returned by the tools. Do not use any outside knowledge about specific \
historical events, dates, places or about any public dataset. If you recognize something, ignore it.
- Information after the forecast start is not available and must not be assumed.
- Typical workflow: inspect the history and reference profiles, check what is special about the \
forecast day (weekday, holiday, weather{news_hint}), then adjust a reference profile accordingly.
- Finish by calling submit_forecast with exactly 24 numbers. Be concise."""


class ForecastEnv:
    def __init__(self, building_id, y_hist, ts_hist, origin, variant="weather",
                 weather_mode="noisy", masked_dates=False):
        if masked_dates and variant == "full":
            raise ValueError("новости требуют настоящих дат: full несовместим с masked_dates")
        self.bid, self.variant = building_id, variant
        self.y = np.asarray(y_hist, float)
        self.ts = pd.DatetimeIndex(ts_hist)
        self.origin = pd.Timestamp(origin)                     # задаётся средой, не агентом
        self.ts_future = pd.date_range(self.origin, periods=24, freq="h")
        self.weather_mode, self.masked = weather_mode, masked_dates
        self.enabled = VARIANT_TOOLS[variant]
        self.done, self.result, self.actions, self.n_invalid = False, None, [], 0

    # ---- служебное
    def specs(self):
        return [{"name": n, **TOOL_SPECS[n]} for n in self.enabled]

    def system_prompt(self):
        day = self._day_label(self.origin)
        prompt = SYSTEM_PROMPT.format(forecast_day=day, history=len(self.y),
                                      news_hint=", news" if "news_search" in self.enabled else "")
        if self.variant == "no_tools":   # данные приходят прямо в сообщении, инструментов анализа нет
            prompt = prompt.replace("returned by the tools", "given in the user message").split(
                "- Typical workflow")[0] + "- Finish by calling submit_forecast with exactly 24 numbers."
        return prompt

    def initial_message(self):
        if self.variant != "no_tools":
            return "Start. Produce the 24-hour forecast."
        rows = "\n".join(f"{self._label(t)}: {v:.4f}" for t, v in zip(self.ts[-168:], self.y[-168:]))
        return (f"Hourly consumption for the last 168 hours (kWh):\n{rows}\n\n"
                f"Forecast {self._day_label(self.origin)}, hours 00-23, and call submit_forecast.")

    def _day_label(self, t):
        t = pd.Timestamp(t)
        if self.masked:
            d = (t.normalize() - self.origin.normalize()).days
            return f"day {d:+d} ({t.strftime('%A')})" if d else f"day 0, the forecast day ({t.strftime('%A')})"
        return t.strftime("%Y-%m-%d (%A)")

    def _label(self, t):
        t = pd.Timestamp(t)
        return f"{self._day_label(t)} {t.strftime('%H:00')}"

    def call(self, name, args):
        self.actions.append({"tool": name, "args": args})
        if name not in self.enabled:
            return json.dumps({"error": f"tool {name} is not available"})
        try:
            out = getattr(self, "_" + name)(**(args or {}))
        except TypeError as e:
            out = {"error": f"bad arguments: {e}"}
        except Exception as e:
            out = {"error": str(e)}
        return json.dumps(out, ensure_ascii=False, default=float)

    # ---- инструменты анализа ряда
    def _profile(self, mask):
        s = pd.Series(self.y, index=self.ts)[mask]
        return [round(float(v), 4) for v in s.groupby(s.index.hour).median().reindex(range(24))]

    def _describe_series(self):
        s = pd.Series(self.y, index=self.ts)
        daily = s.resample("D").sum()
        last7, prev7 = s[-168:].mean(), s[-336:-168].mean()
        mad = (daily - daily.median()).abs().median() * 1.4826
        z = (daily - daily.median()) / (mad + 1e-9)
        unusual = [self._day_label(d) for d, v in z.items() if abs(v) > 3]
        acf = lambda lag: round(float(np.corrcoef(self.y[:-lag], self.y[lag:])[0, 1]), 3)
        return {
            "history_hours": len(self.y), "mean": round(float(s.mean()), 4),
            "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
            "mean_last_7_days": round(float(last7), 4), "mean_previous_7_days": round(float(prev7), 4),
            "trend_last_week_vs_previous_%": round(float((last7 / prev7 - 1) * 100), 1) if prev7 else None,
            "daily_totals": {self._day_label(d): round(float(v), 3) for d, v in daily.items()},
            "median_weekday_profile_by_hour": self._profile(s.index.dayofweek < 5),
            "median_weekend_profile_by_hour": self._profile(s.index.dayofweek >= 5),
            "acf_lag24": acf(24), "acf_lag168": acf(168),
            "unusual_days": unusual,
        }

    def _get_values(self, hours_ago_start, n_hours):
        start = len(self.y) - int(hours_ago_start)
        if not 0 <= start < len(self.y):
            return {"error": f"hours_ago_start must be between 1 and {len(self.y)}"}
        end = min(len(self.y), start + max(1, min(int(n_hours), 72)))  # будущего в среде нет вообще
        return [{"time": self._label(self.ts[i]), "kwh": round(float(self.y[i]), 4)} for i in range(start, end)]

    def _reference_profiles(self):
        r = lambda a: [round(float(v), 4) for v in a]
        s = pd.Series(self.y, index=self.ts)
        return {
            "forecast_day": self._day_label(self.origin),
            "yesterday": r(self.y[-24:]),
            "same_weekday_last_week": r(self.y[-168:-144]),
            "mean_same_weekday_last_2_weeks": r((self.y[-168:-144] + self.y[-336:-312]) / 2),
            "median_weekday_profile": self._profile(s.index.dayofweek < 5),
            "median_weekend_profile": self._profile(s.index.dayofweek >= 5),
        }

    # ---- контекст
    def _calendar(self):
        days = pd.date_range(self.ts[0].normalize(), self.origin.normalize(), freq="D")
        out = []
        for d in days:
            h = context.holiday_name(self.bid, d)
            item = {"day": self._day_label(d), "weekday": d.strftime("%A"),
                    "public_holiday": bool(h)}
            if h and not self.masked:
                item["holiday_name"] = h
            out.append(item)
        return {"forecast_day": out[-1], "history_days": out[:-1]}

    def _building_info(self):
        info = context.building_info(self.bid, reveal_city=(self.variant == "full"))
        if self.masked:
            info.pop("timezone", None)
        return info

    def _weather(self):
        hist, fut = context.weather(self.bid, self.ts, self.ts_future, self.weather_mode,
                                    key=(self.bid, str(self.origin)))
        daily = hist.groupby(hist.index.normalize()).agg(["min", "mean", "max"]).round(1)
        out = {"history_daily_temperature_C": {self._day_label(d): row.to_dict() for d, row in daily.iterrows()}}
        if fut is None:
            out["forecast_day"] = "temperature forecast not available"
        else:
            out["forecast_day_hourly_temperature_forecast_C"] = [round(float(v), 1) for v in fut]
        return out

    def _news_search(self, date, query=None):
        return context.news(date, self.origin, query, utc_offset=context.SITES[context.site_of(self.bid)]["utc_offset"])

    # ---- финал
    def _submit_forecast(self, values, reasoning=""):
        v = np.asarray(values, float)
        if v.shape != (24,) or not np.isfinite(v).all() or (v < 0).any():
            self.n_invalid += 1
            return {"error": "values must be exactly 24 finite non-negative numbers"}
        self.done = True
        self.result = {"pred": v, "reasoning": reasoning}
        return {"ok": True}
