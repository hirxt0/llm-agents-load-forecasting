import numpy as np

def forecast_metrics(y_true, y_pred, history):
    y_true, y_pred, h = (np.asarray(a, float) for a in (y_true, y_pred, history))
    err = y_pred - y_true
    scale = h.mean()
    naive = np.mean(np.abs(h[24:] - h[:-24]))
    mae, rmse = (np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2)))
    return {'NRMSE': float(rmse / scale * 100), 'NMAE': float(mae / scale * 100), 'NMBE': float(err.mean() / scale * 100), 'MASE': float(mae / naive) if naive > 0 else float('nan'), 'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-09)) * 100)}
