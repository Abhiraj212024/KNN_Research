# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:39:59.139186Z","iopub.execute_input":"2025-11-11T20:39:59.139997Z","iopub.status.idle":"2025-11-11T20:40:20.164788Z","shell.execute_reply.started":"2025-11-11T20:39:59.139970Z","shell.execute_reply":"2025-11-11T20:40:20.164100Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import gc

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:40:20.165891Z","iopub.execute_input":"2025-11-11T20:40:20.166556Z","iopub.status.idle":"2025-11-11T20:41:06.538223Z","shell.execute_reply.started":"2025-11-11T20:40:20.166536Z","shell.execute_reply":"2025-11-11T20:41:06.537520Z"},"jupyter":{"outputs_hidden":false}}
dataframes = []
for i in range(0, 31):
  PATH = f"/kaggle/input/ebx-tick/day{i}.parquet"

  try:
    data = pd.read_parquet(PATH)
    dataframes.append(data)
    if i % 10 == 0:
        print(f"Successfully loaded data upto day {i}")
  except:
    print(f"Error loading data from day {i}.")

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.539102Z","iopub.execute_input":"2025-11-11T20:41:06.539435Z","iopub.status.idle":"2025-11-11T20:41:06.590089Z","shell.execute_reply.started":"2025-11-11T20:41:06.539406Z","shell.execute_reply":"2025-11-11T20:41:06.589471Z"},"jupyter":{"outputs_hidden":false}}
importance_1 = pd.read_csv("/kaggle/input/feature-importance-ebx/feature_importance_of_EBX_BB.csv")
importance_2 = pd.read_csv("/kaggle/input/feature-importance-ebx/feature_importance_of_EBX_PB.csv")
importance_3 = pd.read_csv("/kaggle/input/feature-importance-ebx/feature_importance_of_EBX_PV.csv")
importance_4 = pd.read_csv("/kaggle/input/feature-importance-ebx/feature_importance_of_EBX_V_VB.csv")

l1 = ((importance_1.sort_values(by= 'Importance_rank_score', ascending= False))['Feature']).iloc[0 : 6].to_list()
l2 = ((importance_2.sort_values(by= 'Importance_rank_score', ascending= False))['Feature']).iloc[0 : 6].to_list()
l3 = ((importance_3.sort_values(by= 'Importance_rank_score', ascending= False))['Feature']).iloc[0 : 6].to_list()
l4 = ((importance_4.sort_values(by= 'Importance_rank_score', ascending= False))['Feature']).iloc[0 : 6].to_list()

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.591678Z","iopub.execute_input":"2025-11-11T20:41:06.592179Z","iopub.status.idle":"2025-11-11T20:41:06.597324Z","shell.execute_reply.started":"2025-11-11T20:41:06.592154Z","shell.execute_reply":"2025-11-11T20:41:06.596602Z"},"jupyter":{"outputs_hidden":false}}
# Include all feature families, but EXCLUDE Price (since it's the target)
# regex = r'^((PB|VB|V|PV|BB)\d+.*_T\d+)$|^((PB|VB|V|PV|BB)\d+)$'

feature_cols = list(set(l1).union(set(l2), set(l3), set(l4)))

# Double-check Price is not included
feature_cols = [c for c in feature_cols if c != 'Price']

print(f"Total features: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")
print(f"Price in features: {'Price' in feature_cols}")

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.598075Z","iopub.execute_input":"2025-11-11T20:41:06.598315Z","iopub.status.idle":"2025-11-11T20:41:06.612918Z","shell.execute_reply.started":"2025-11-11T20:41:06.598291Z","shell.execute_reply":"2025-11-11T20:41:06.612109Z"},"jupyter":{"outputs_hidden":false}}
all_metrics = []

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.613761Z","iopub.execute_input":"2025-11-11T20:41:06.614020Z","iopub.status.idle":"2025-11-11T20:41:06.628508Z","shell.execute_reply.started":"2025-11-11T20:41:06.613996Z","shell.execute_reply":"2025-11-11T20:41:06.627864Z"},"jupyter":{"outputs_hidden":false}}
# ------------------------------------------------------------
# ✨ CHANGE: Deterministic feature space builder (no probing)
# ------------------------------------------------------------
def build_feature_space(
    base_feature_cols: list[str],
    stats=('mean','median','std','min','max'),
    roll_windows=(5,10),
    price_col='Price'
) -> list[str]:
    """
    Returns the canonical engineered feature list (no raw inputs).
    For each base feature we include intra-bar stats and rolling (5/10) for mean/median.
    Also includes price-derived features.
    """
    engineered = []

    # Intra-bar stats for each base feature at 10s resolution
    for col in base_feature_cols:
        for s in stats:
            engineered.append(f"{col}_{s}")

    # Rolling over bars ONLY for mean/median (keeps width reasonable)
    for col in base_feature_cols:
        for s in ('mean','median'):
            for w in roll_windows:
                engineered.append(f"{col}_{s}_roll{w}")

    # Price-derived features (from OHLC of price_col)
    price_feats = [
        "Return_1",              # 1-bar return (10s)
        "Return_roll5_mean",
        "Return_roll10_mean",
        "Return_roll5_std",
        "Return_roll10_std",
        "HL_spread",            # (High-Low)/Close (per bar)
        "OC_change",            # (Close-Open)/Open
        "Cumulative_Return"     # from first bar of the day
    ]
    engineered.extend(price_feats)

    return engineered

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.629278Z","iopub.execute_input":"2025-11-11T20:41:06.629506Z","iopub.status.idle":"2025-11-11T20:41:06.647764Z","shell.execute_reply.started":"2025-11-11T20:41:06.629481Z","shell.execute_reply":"2025-11-11T20:41:06.647061Z"},"jupyter":{"outputs_hidden":false}}
def aggregate_to_10s_bars(df, raw_feature_cols, price_col="Price"):
    """
    Convert 1-second tick data into aggregated 10-second bars.
    Produces compact statistical summaries for each raw feature.
    This reduces feature explosion but preserves intraperiod information.
    """
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df["Time_10s"] = df["Time"].dt.floor("10s")

    # Initialize aggregation dict
    agg_dict = {}

    # OHLC from proxy price
    agg_dict[price_col] = ["first", "max", "min", "last"]

    # Compact summaries for each raw feature (OPTION A)
    stats = ["mean", "std", "min", "max", "median"]
    for col in raw_feature_cols:
        if col == price_col:
            continue
        # For each feature:
        # X_mean_10s , X_std_10s , X_min_10s , X_max_10s , X_median_10s
        agg_dict[col] = stats

    # Perform groupby aggregation
    df_agg = df.groupby("Time_10s").agg(agg_dict)

    # Flatten MultiIndex columns
    df_agg.columns = [
        f"{c[0]}_{c[1]}_10s" if c[1] != "" else c[0]
        for c in df_agg.columns
    ]

    df_agg = df_agg.reset_index()

    # Rename proxy OHLC
    rename_map = {
        f"{price_col}_first_10s": "Open",
        f"{price_col}_max_10s": "High",
        f"{price_col}_min_10s": "Low",
        f"{price_col}_last_10s": "Close",
    }
    df_agg.rename(columns=rename_map, inplace=True)

    # Sanity checks
    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df_agg.columns:
            raise ValueError(f"Missing OHLC column: {col}")

    # Basic price-derived features
    df_agg["Return"] = df_agg["Close"].pct_change()
    df_agg["HL_spread"] = (df_agg["High"] - df_agg["Low"]) / df_agg["Close"]
    df_agg["OC_change"] = (df_agg["Close"] - df_agg["Open"]) / df_agg["Open"]

    df_agg = df_agg.dropna().reset_index(drop=True)
    return df_agg

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.648504Z","iopub.execute_input":"2025-11-11T20:41:06.648756Z","iopub.status.idle":"2025-11-11T20:41:06.665635Z","shell.execute_reply.started":"2025-11-11T20:41:06.648739Z","shell.execute_reply":"2025-11-11T20:41:06.664900Z"},"jupyter":{"outputs_hidden":false}}
def engineer_features_10s(df):
    """
    Create compact engineered features from 10s bars.
    Uses only OHLC + Return + basic rolling stats.
    """
    df = df.copy()

    # Rolling returns: short-term structure
    df["Return_MA_3"] = df["Return"].rolling(3).mean()
    df["Return_MA_6"] = df["Return"].rolling(6).mean()
    df["Return_STD_3"] = df["Return"].rolling(3).std()
    df["Return_STD_6"] = df["Return"].rolling(6).std()

    # Trend features
    df["Price_Trend_3"] = (df["Close"] - df["Close"].shift(3)) / df["Close"].shift(3)
    df["Price_Trend_6"] = (df["Close"] - df["Close"].shift(6)) / df["Close"].shift(6)

    # Acceleration
    df["Return_Accel"] = df["Return"] - df["Return"].shift(1)

    # Normalized spreads
    df["HL_spread_norm"] = df["HL_spread"] / df["HL_spread"].rolling(20).mean()
    df["OC_change_norm"] = df["OC_change"] / df["OC_change"].rolling(20).mean()

    # Candlestick structure
    df["Candle_Body"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Close_vs_HL"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-6)

    # Autoregressive lags
    for lag in [1, 2, 3]:
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    # Cumulative trend from day's start
    df["Cumulative_Return"] = df["Close"] / df["Close"].iloc[0] - 1

    df = df.dropna().reset_index(drop=True)
    return df

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.666441Z","iopub.execute_input":"2025-11-11T20:41:06.666712Z","iopub.status.idle":"2025-11-11T20:41:06.684930Z","shell.execute_reply.started":"2025-11-11T20:41:06.666689Z","shell.execute_reply":"2025-11-11T20:41:06.684195Z"},"jupyter":{"outputs_hidden":false}}
def compute_canonical_features(dataframes, raw_feature_cols, probe_days=5):
    """
    Determine a consistent feature list across days.
    Looks at aggregated + engineered features.
    Only keeps numeric columns.
    """
    feature_sets = []

    for i in range(min(probe_days, len(dataframes))):
        df_raw = dataframes[i]

        # Apply new feature pipeline
        df_10s = aggregate_to_10s_bars(df_raw, raw_feature_cols)
        eng = engineer_features_10s(df_10s)

        numeric_cols = [
            c for c in eng.columns
            if c != "Close" and pd.api.types.is_numeric_dtype(eng[c])
        ]

        feature_sets.append(set(numeric_cols))

    canonical = sorted(set().union(*feature_sets))

    print(f"[Canonical Feature Count] {len(canonical)}")
    return canonical

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.687239Z","iopub.execute_input":"2025-11-11T20:41:06.687444Z","iopub.status.idle":"2025-11-11T20:41:06.705588Z","shell.execute_reply.started":"2025-11-11T20:41:06.687429Z","shell.execute_reply":"2025-11-11T20:41:06.704994Z"},"jupyter":{"outputs_hidden":false}}
def prepare_day_data(df_eng: pd.DataFrame, feature_cols: list[str], lookback: int, horizons: list[int]):
    """
    Prepare sequences for ONE day from engineered feature frame.
    """
    dfX = df_eng[feature_cols].astype(np.float32)
    close = df_eng['Close'].astype(np.float32)

    n_samples = len(dfX) - lookback - max(horizons)
    if n_samples <= 0: 
        return None, None

    X = np.zeros((n_samples, lookback, len(feature_cols)), dtype=np.float32)
    y = {h: np.zeros(n_samples, dtype=np.float32) for h in horizons}

    for i in range(n_samples):
        j = i + lookback
        X[i] = dfX.iloc[i:j].values
        cur = close.iloc[j]
        for h in horizons:
            fut = close.iloc[j + h]
            y[h][i] = (fut - cur) / (cur + 1e-12)

    return X, y


def prepare_all_days(dataframes: list[pd.DataFrame],
                     base_feature_cols: list[str],
                     lookback: int,
                     horizons: list[int],
                     price_col: str = 'Price') -> tuple[list[dict], list[str]]:
    """
    Aggregates → Engineers → Sequences for all days.
    Returns:
      - list of dicts per day: {'day_idx','X','y','n_samples'}
      - canonical engineered feature list
    """
    # ✨ CHANGE: canonical engineered names are deterministic from base features
    engineered_cols = build_feature_space(base_feature_cols, price_col=price_col)

    all_day_data = []
    for idx, df in enumerate(dataframes):
        bars = aggregate_to_10s_bars(df, base_feature_cols, price_col=price_col)
        eng = engineer_features_10s(bars)

        # Reindex to canonical columns
        eng = eng.reindex(columns=engineered_cols + ['Close'])
        eng = eng.ffill().bfill().fillna(0.0)

        X, y = prepare_day_data(eng, engineered_cols, lookback, horizons)
        if X is None:
            print(f"Day {idx}: insufficient bars → skipped")
            continue

        all_day_data.append({
            'day_idx': idx,
            'X': X,
            'y': y,
            'n_samples': len(X)
        })
        print(f"Day {idx}: {len(df)} ticks → {len(bars)} bars → {len(X)} samples")
        print("Number of engineered_columns:", len(engineered_cols))

    return all_day_data, engineered_cols

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.706310Z","iopub.execute_input":"2025-11-11T20:41:06.706519Z","iopub.status.idle":"2025-11-11T20:41:06.726065Z","shell.execute_reply.started":"2025-11-11T20:41:06.706497Z","shell.execute_reply":"2025-11-11T20:41:06.725426Z"},"jupyter":{"outputs_hidden":false}}
class PerDayBatchGeneratorDual(tf.keras.utils.Sequence):
    
    def __init__(self, day_data_list, batch_size, horizons, shuffle=True, augment=False):
        self.day_data_list = day_data_list
        self.batch_size = batch_size
        self.horizons = horizons
        self.shuffle = shuffle
        self.augment = augment
        
        self.day_indices = []
        for day_data in day_data_list:
            day_idx = day_data['day_idx']
            n_samples = day_data['n_samples']
            for i in range(n_samples):
                self.day_indices.append((day_idx, i))

        self.total_samples = len(self.day_indices)
        
        if self.shuffle:
            np.random.shuffle(self.day_indices)

    def __len__(self):
        return int(np.ceil(self.total_samples / self.batch_size))

    def __getitem__(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.total_samples)
        batch_day_indices = self.day_indices[start_idx:end_idx]

        X_batch = []
        y_batches = {}
        for h in self.horizons:
            y_batches[f'reg_{h}'] = []
            y_batches[f'cls_{h}'] = []

        # build batch
        for day_idx, sample_idx in batch_day_indices:
            day_data = next(d for d in self.day_data_list if d['day_idx'] == day_idx)
            X = day_data['X'][sample_idx]

            if self.augment:
                noise = np.random.normal(0, 0.01, X.shape)
                X = X + noise

            X_batch.append(X)

            for h in self.horizons:
                ret = day_data['y'][h][sample_idx]

                y_batches[f'reg_{h}'].append(ret)
                y_batches[f'cls_{h}'].append(1.0 if ret > 0 else 0.0)

                # ✅ Large-move boosting
                abs_ret = abs(ret)
                # We compute 85th percentile ONCE per day for each horizon
                threshold = np.quantile(np.abs(day_data['y'][h]), 0.85)

                if abs_ret >= threshold:
                    w = 2.0           # double weight for big returns
                else:
                    w = 1.0


        X_batch = np.array(X_batch, dtype=np.float32)
        y_dict = {k: np.array(v, dtype=np.float32) for k, v in y_batches.items()}

        return X_batch, y_dict     

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.day_indices)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.726706Z","iopub.execute_input":"2025-11-11T20:41:06.726880Z","iopub.status.idle":"2025-11-11T20:41:06.749070Z","shell.execute_reply.started":"2025-11-11T20:41:06.726866Z","shell.execute_reply":"2025-11-11T20:41:06.748528Z"},"jupyter":{"outputs_hidden":false}}
# ---------- NEW: FeatureMixer ----------
from tensorflow.keras import layers, models, activations

class FeatureMixer(layers.Layer):
    """
    Residual 1x1 Conv feature-mixing block with PreNorm.
    - Normalizes per timestep across features
    - Mixes features with 1x1 Conv, GELU, Dropout, 1x1 Conv
    - Residual add
    """
    def __init__(self, proj_dim=None, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.proj1 = None
        self.act = layers.Activation("gelu")
        self.drop = layers.Dropout(dropout)
        self.proj2 = None

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        d_mid = self.proj_dim or d_in
        self.proj1 = layers.Conv1D(d_mid, kernel_size=1, use_bias=False)
        self.proj2 = layers.Conv1D(d_in,  kernel_size=1, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.norm(x)
        h = self.proj1(h)
        h = self.act(h)
        h = self.drop(h, training=training)
        h = self.proj2(h)
        return layers.Add()([x, h])


# ---------- UPDATED: TCN ----------
def build_tcn_model_dual(input_shape, horizons=(6, 12), lr=1e-3, dropout=0.3):
    """
    Causal TCN backbone + dual heads (reg + cls) per horizon.
    Uses FeatureMixer upfront (no external scaling needed).
    """
    inp = layers.Input(shape=input_shape)  # (seq_len, n_features)

    x = FeatureMixer(dropout=dropout)(inp)

    def tcn_block(x, filters, dilation, dropout):
        # PreNorm
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.Conv1D(filters, kernel_size=3, padding="causal",
                          dilation_rate=dilation)(h)
        h = layers.Activation("gelu")(h)
        h = layers.Dropout(dropout)(h)
        h = layers.Conv1D(filters, kernel_size=3, padding="causal",
                          dilation_rate=dilation)(h)
        h = layers.Activation("gelu")(h)
        # Match channels for residual if needed
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, kernel_size=1, padding="same")(x)
        return layers.Add()([x, h])

    # Hierarchy of dilations
    x = tcn_block(x, filters=64, dilation=1, dropout=dropout)
    x = tcn_block(x, filters=64, dilation=2, dropout=dropout)
    x = tcn_block(x, filters=64, dilation=4, dropout=dropout)
    x = tcn_block(x, filters=32, dilation=8, dropout=dropout)

    x = layers.GlobalAveragePooling1D()(x)

    shared = layers.LayerNormalization(epsilon=1e-6)(x)
    shared = layers.Dense(128, activation="gelu")(shared)
    shared = layers.Dropout(dropout)(shared)
    shared = layers.Dense(64, activation="gelu")(shared)

    outputs = []
    for h in horizons:
        head = layers.LayerNormalization(epsilon=1e-6)(shared)
        head = layers.Dense(32, activation="gelu", name=f"base_{h}")(head)
        head = layers.Dropout(dropout * 0.5)(head)

        reg = layers.Dense(1, name=f"reg_{h}")(head)
        cls = layers.Dense(1, activation="sigmoid", name=f"cls_{h}")(head)
        outputs += [reg, cls]

    model = models.Model(inp, outputs, name="tcn_dual_featmix")
    return model


# ---------- UPDATED: GRU ----------
def build_gru_model_dual(input_shape, horizons=[18, 24], lr=1e-3, dropout=0.3):
    """
    Upgraded GRU dual-head model with residual connections,
    attention pooling, and better normalization.
    """
    inp = layers.Input(shape=input_shape)

    # 🌟 Layer normalization + input mixing
    x = layers.LayerNormalization()(inp)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.SpatialDropout1D(0.1)(x)

    # 🌊 First GRU block
    x1 = layers.Bidirectional(
        layers.GRU(256, return_sequences=True, dropout=dropout, recurrent_dropout=0.1)
    )(x)

    # 🔁 Second GRU block (residual)
    x2 = layers.Bidirectional(
        layers.GRU(192, return_sequences=True, dropout=dropout, recurrent_dropout=0.1)
    )(x1)
    x2 = layers.Add()([x1, x2])  # Residual connection

    # ⚡ Third GRU block
    x3 = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=dropout, recurrent_dropout=0.1)
    )(x2)
    x3 = layers.Add()([x2, x3])

    # 🔍 Attention pooling (context-aware summary)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)
    context = attn(x3, x3)
    context = layers.Add()([x3, context])

    # 🧠 Global context vector
    x = layers.LayerNormalization()(context)
    x = layers.GlobalAveragePooling1D()(x)

    # 🔩 Dense projection
    shared = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(64, activation='relu')(shared)

    # 🎯 Multi-horizon outputs
    outputs = []
    for h in horizons:
        base = layers.Dense(32, activation='relu')(shared)
        base = layers.Dropout(0.2)(base)

        reg = layers.Dense(1, name=f"reg_{h}")(base)
        cls = layers.Dense(1, activation='sigmoid', name=f"cls_{h}")(base)

        outputs.extend([reg, cls])

    model = keras.Model(inp, outputs, name="GRU_dual_enhanced")


    return model

# ---------- UPDATED: Transformer ----------
def build_transformer_model_dual(input_shape, horizons=(30,), lr=5e-4, dropout=0.3, d_model=None, n_heads=4, n_blocks=2):
    """
    Lightweight Transformer encoder + dual heads per horizon.
    - Starts with FeatureMixer
    - Projects to d_model with 1x1 Conv
    - PreNorm Attention + FFN blocks
    """
    seq_len, d_in = input_shape
    d_model = d_model or max(64, min(256, d_in))  # sensible default

    inp = layers.Input(shape=input_shape)

    x = FeatureMixer(dropout=dropout)(inp)
    x = layers.Conv1D(d_model, kernel_size=1, use_bias=False)(x)  # channel proj

    # Learned positional embeddings (fixed seq_len assumed in training)
    positions = layers.Embedding(input_dim=seq_len, output_dim=d_model)(
        tf.range(seq_len)
    )
    positions = tf.expand_dims(positions, 0)  # (1, seq_len, d_model)
    x = layers.Add()([x, positions])

    def encoder_block(x):
        # PreNorm Self-Attention
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads,
                                      dropout=dropout)(h, h)
        h = layers.Dropout(dropout)(h)
        x = layers.Add()([x, h])

        # PreNorm FFN
        h2 = layers.LayerNormalization(epsilon=1e-6)(x)
        h2 = layers.Dense(d_model * 4, activation="gelu")(h2)
        h2 = layers.Dropout(dropout)(h2)
        h2 = layers.Dense(d_model)(h2)
        h2 = layers.Dropout(dropout)(h2)
        return layers.Add()([x, h2])

    for _ in range(n_blocks):
        x = encoder_block(x)

    x = layers.GlobalAveragePooling1D()(x)

    shared = layers.LayerNormalization(epsilon=1e-6)(x)
    shared = layers.Dense(128, activation="gelu")(shared)
    shared = layers.Dropout(dropout)(shared)
    shared = layers.Dense(64, activation="gelu")(shared)

    outputs = []
    for h in horizons:
        head = layers.LayerNormalization(epsilon=1e-6)(shared)
        head = layers.Dense(32, activation="gelu", name=f"base_{h}")(head)
        head = layers.Dropout(dropout * 0.5)(head)

        reg = layers.Dense(1, name=f"reg_{h}")(head)
        cls = layers.Dense(1, activation="sigmoid", name=f"cls_{h}")(head)
        outputs += [reg, cls]

    model = models.Model(inp, outputs, name="transformer_dual_featmix")
    return model

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.749891Z","iopub.execute_input":"2025-11-11T20:41:06.750204Z","iopub.status.idle":"2025-11-11T20:41:06.769887Z","shell.execute_reply.started":"2025-11-11T20:41:06.750176Z","shell.execute_reply":"2025-11-11T20:41:06.769297Z"},"jupyter":{"outputs_hidden":false}}
# ============================================================================
# MODIFIED TRAINING FUNCTION
# ============================================================================

def train_model_dual_for_horizons(train_day_data, val_day_data, feature_cols,
                                   horizons, model_type='tcn', lookback=18):
    """
    Train a dual-head model (regression + classification) for specific horizons.
    """
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} DUAL-HEAD for horizons {horizons}")
    print(f"{'='*70}")
    
    # Create generators
    train_gen = PerDayBatchGeneratorDual(
        train_day_data, batch_size=256, horizons=horizons,
        shuffle=True, augment=True
    )
    
    val_gen = PerDayBatchGeneratorDual(
        val_day_data, batch_size=256, horizons=horizons,
        shuffle=False, augment=False
    )
    
    # Build model
    input_shape = (lookback, len(feature_cols))
    
    if model_type == 'tcn':
        model = build_tcn_model_dual(input_shape, horizons, lr=1e-3, dropout=0.3)
    elif model_type == 'gru':
        model = build_gru_model_dual(input_shape, horizons, lr=5e-4, dropout=0.3)
    elif model_type == 'transformer':
        model = build_transformer_model_dual(input_shape, horizons, lr=5e-4, dropout=0.3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("\nModel Summary:")
    model.summary()
    
    # Compile with dual losses
    loss_dict = {}
    loss_weights = {}
    metrics_dict = {}
    
    for h in horizons:
        # Regression head: reduce weight slightly
        loss_dict[f'reg_{h}'] = keras.losses.Huber(delta=1.0)
        loss_weights[f'reg_{h}'] = 1.0
        metrics_dict[f'reg_{h}'] = ['mae']
    
        # Classification head: emphasize directional learning
        loss_dict[f'cls_{h}'] = tf.losses.BinaryCrossentropy(label_smoothing=0.05)
        loss_weights[f'cls_{h}'] = 3.0
        metrics_dict[f'cls_{h}'] = ['accuracy']
    
    opt = optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    
    model.compile(
        optimizer=opt,
        loss=loss_dict,
        loss_weights=loss_weights,
        metrics=metrics_dict
    )
    
    # Train
    from tensorflow.keras import callbacks

    cb_early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=1e-5,
        verbose=1
    )

    cb_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=4,
        min_lr=1e-6,
        cooldown=2,
        verbose=1
    )

    cb_chk = keras.callbacks.ModelCheckpoint(
        filepath=f"best_{model_type}_dual.keras",
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )

    print("\n=== MODEL OUTPUT NAMES ===")
    print(model.output_names)
    
    print("\n=== LOSS DICT KEYS ===")
    print(list(loss_dict.keys()))
    
    print("\n=== GENERATOR OUTPUT KEYS ===")
    batch_X, batch_y = train_gen[0]
    print(batch_y.keys())


    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=40,
        callbacks=[cb_early, cb_reduce, cb_chk],
        verbose=1,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen)
   )

    
    return model, history

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.770578Z","iopub.execute_input":"2025-11-11T20:41:06.771239Z","iopub.status.idle":"2025-11-11T20:41:06.789631Z","shell.execute_reply.started":"2025-11-11T20:41:06.771223Z","shell.execute_reply":"2025-11-11T20:41:06.788896Z"},"jupyter":{"source_hidden":true,"outputs_hidden":false}}
# ============================================================================
# PART 7: EVALUATION (Per-Day Metrics)
# ============================================================================

def evaluate_model_per_day(model, day_data_list, horizons):
    """
    Evaluate model on each day separately and aggregate results.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating model on {len(day_data_list)} days")
    print(f"{'='*70}")
    
    all_results = []
    EPS= 1e-6
    
    for day_data in day_data_list:
        day_idx = day_data['day_idx']
        X = day_data['X']
        y_true_dict = day_data['y']
        
        # Predict
        predictions = model.predict(X, batch_size=512, verbose=0)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Calculate metrics for each horizon
        day_metrics = {'day_idx': day_idx, 'n_samples': len(X)}
        
        for i, h in enumerate(horizons):
            y_true = y_true_dict[h]
            y_pred = predictions[i].flatten()
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # Directional accuracy
            direction_correct = np.mean(
                ((y_true >  EPS) & (y_pred > 0)) |
                ((y_true < -EPS) & (y_pred < 0)) |
                ((np.abs(y_true) <= EPS) & (np.abs(y_pred) <= EPS))
            )
            
            day_metrics[f'h{h}_mae'] = mae
            day_metrics[f'h{h}_rmse'] = rmse
            day_metrics[f'h{h}_r2'] = r2
            day_metrics[f'h{h}_dir_acc'] = direction_correct
            
            print(f"Day {day_idx} | Horizon {h} ({h*10}s) → "
                  f"MAE={mae:.6f} ({mae*100:.4f}%), RMSE={rmse:.6f}, "
                  f"R²={r2:.4f}, Dir_Acc={direction_correct:.3f}")
        
        all_results.append(day_metrics)
    
    # Aggregate across all days
    print(f"\n{'-'*70}")
    print("AVERAGE ACROSS ALL DAYS:")
    print(f"{'-'*70}")
    
    for h in horizons:
        avg_mae = np.mean([r[f'h{h}_mae'] for r in all_results])
        avg_r2 = np.mean([r[f'h{h}_r2'] for r in all_results])
        avg_dir = np.mean([r[f'h{h}_dir_acc'] for r in all_results])
        
        print(f"Horizon {h} ({h*10}s) → "
              f"Avg MAE={avg_mae:.6f} ({avg_mae*100:.4f}%), "
              f"Avg R²={avg_r2:.4f}, Avg Dir_Acc={avg_dir:.3f}")
    
    return all_results

# %% [code] {"jupyter":{"source_hidden":true,"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-11T20:41:06.790429Z","iopub.execute_input":"2025-11-11T20:41:06.790608Z","iopub.status.idle":"2025-11-11T20:41:06.809439Z","shell.execute_reply.started":"2025-11-11T20:41:06.790594Z","shell.execute_reply":"2025-11-11T20:41:06.808693Z"}}

def evaluate_dual_model_quant(model, day_data_list, horizons):
    """
    Evaluate dual-head model with special focus on:
    1. Pure directional accuracy from classification head
    2. Combined accuracy (using both regression and classification)
    """
    print("\n" + "="*70)
    print(f" DUAL-HEAD QUANT EVALUATION ON {len(day_data_list)} DAYS")
    print("="*70)
    
    all_results = []
    
    for day_data in day_data_list:
        X = day_data["X"]
        y_true_dict = day_data["y"]
        day_idx = day_data["day_idx"]
        
        preds = model.predict(X, batch_size=512, verbose=0)
        if not isinstance(preds, list):
            preds = [preds]
        
        day_metrics = {"day_idx": day_idx}
        
        print(f"\n---- Day {day_idx} ----")
        
        # Predictions come in pairs: [reg_6, cls_6, reg_12, cls_12, ...]
        for i, h in enumerate(horizons):
            reg_idx = i * 2
            cls_idx = i * 2 + 1
            
            y_true = y_true_dict[h]
            y_pred_reg = preds[reg_idx].flatten()
            y_pred_cls = preds[cls_idx].flatten()  # Probability of positive
            
            # Convert classification probabilities to binary predictions
            y_pred_dir_cls = (y_pred_cls > 0.5).astype(float)  # 1 or 0
            y_true_dir = (y_true > 0).astype(float)
            
            # 1. Pure classification accuracy
            cls_accuracy = np.mean(y_pred_dir_cls == y_true_dir)
            
            # 2. Regression-based directional accuracy
            reg_da = np.mean(np.sign(y_pred_reg) == np.sign(y_true))
            
            # 3. Combined approach: use classification for direction, regression for magnitude
            y_pred_combined = np.where(y_pred_cls > 0.5, 
                                       np.abs(y_pred_reg), 
                                       -np.abs(y_pred_reg))
            combined_da = np.mean(np.sign(y_pred_combined) == np.sign(y_true))
            
            # 4. Accuracy on large moves
            threshold = np.quantile(np.abs(y_true), 0.75)
            mask_big = np.abs(y_true) >= threshold
            da_big = np.mean(np.sign(y_pred_combined[mask_big]) == np.sign(y_true[mask_big]))
            
            # 5. Correlation
            corr = np.corrcoef(y_pred_combined, y_true)[0, 1] if np.std(y_pred_combined) > 0 else 0
            
            # 6. PnL and Sharpe
            pnl = y_pred_combined * y_true
            pnl_mean = pnl.mean()
            pnl_std = pnl.std() + 1e-8
            sharpe = pnl_mean / pnl_std
            
            # print(f"Horizon {h} ({h*10}s)")
            # print(f"  Classification Head Accuracy:   {cls_accuracy:.4f}")
            # print(f"  Regression DA:                  {reg_da:.4f}")
            # print(f"  Combined DA:                    {combined_da:.4f}")
            # print(f"  DA on large moves:              {da_big:.4f}")
            # print(f"  Correlation:                    {corr:.4f}")
            # print(f"  Sharpe:                         {sharpe:.4f}")
            
            # Save metrics
            day_metrics[f"h{h}_cls_acc"] = cls_accuracy
            day_metrics[f"h{h}_reg_da"] = reg_da
            day_metrics[f"h{h}_combined_da"] = combined_da
            day_metrics[f"h{h}_da_big"] = da_big
            day_metrics[f"h{h}_corr"] = corr
            day_metrics[f"h{h}_sharpe"] = sharpe

            all_metrics.append({
                "day": day_idx,
                "horizon": h,
                "cls_acc": cls_accuracy,
                "reg_da": reg_da,
                "combined_da": combined_da,
                "big_move_da": da_big,
                "corr": corr,
                "sharpe": sharpe,
                "model_type": model.name
            })
                    
        all_results.append(day_metrics)
    
    # Aggregate metrics
    print("\n" + "-"*70)
    print("AGGREGATE DUAL-HEAD METRICS")
    print("-"*70)
    
    for h in horizons:
        cls_acc = np.mean([r[f"h{h}_cls_acc"] for r in all_results])
        reg_da = np.mean([r[f"h{h}_reg_da"] for r in all_results])
        combined_da = np.mean([r[f"h{h}_combined_da"] for r in all_results])
        da_big = np.mean([r[f"h{h}_da_big"] for r in all_results])
        corr = np.mean([r[f"h{h}_corr"] for r in all_results])
        sharpe = np.mean([r[f"h{h}_sharpe"] for r in all_results])
        
        print(f"\nHorizon {h} ({h*10}s)")
        print(f"  Avg Classification Accuracy:   {cls_acc:.4f}")
        print(f"  Avg Regression DA:             {reg_da:.4f}")
        print(f"  Avg Combined DA:               {combined_da:.4f}")
        print(f"  Avg DA on large moves:         {da_big:.4f}")
        print(f"  Avg Correlation:               {corr:.4f}")
        print(f"  Avg Sharpe:                    {sharpe:.4f}")
    
    return all_results

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.810257Z","iopub.execute_input":"2025-11-11T20:41:06.810463Z","iopub.status.idle":"2025-11-11T20:41:06.827965Z","shell.execute_reply.started":"2025-11-11T20:41:06.810448Z","shell.execute_reply":"2025-11-11T20:41:06.827145Z"},"jupyter":{"outputs_hidden":false}}
def complete_training_pipeline_dual(
    dataframes: list[pd.DataFrame],
    base_feature_cols: list[str],
    train_ratio: float = 0.8,
    lookback: int = 18,
    price_col: str = 'Price'
):
    horizons_cfg = {
        'tcn': [6, 12],        # 1m, 2m
        'gru': [18, 24],       # 3m, 4m
        'transformer': [30]    # 5m
    }
    all_horizons = [6,12,18,24,30]

    print("\n==== Step 1: Prepare all days ====")
    all_days, engineered_cols = prepare_all_days(
        dataframes, base_feature_cols, lookback, all_horizons, price_col=price_col
    )

    n_train = int(len(all_days) * train_ratio)
    train_days = all_days[:n_train]
    val_days   = all_days[n_train:]
    print(f"Train days: {len(train_days)} | Val days: {len(val_days)}")
    results = {}

    for model_type, horizons in horizons_cfg.items():
        model, hist = train_model_dual_for_horizons(train_days, val_days, engineered_cols, horizons, model_type, lookback)
        eval_res = evaluate_dual_model_quant(model, val_days, horizons)
        results[model_type] = {'model': model, 'history': hist, 'eval': eval_res, 'horizons': horizons}
        model.save(f"{model_type}_dual_model.keras")
        gc.collect()

    return results, engineered_cols

# %% [code] {"jupyter":{"source_hidden":true,"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-11T20:41:06.828710Z","iopub.execute_input":"2025-11-11T20:41:06.828909Z","iopub.status.idle":"2025-11-11T20:41:06.842076Z","shell.execute_reply.started":"2025-11-11T20:41:06.828895Z","shell.execute_reply":"2025-11-11T20:41:06.841280Z"}}
def plot_metric_boxplot(metrics_df, metric, title=None):
    plt.figure(figsize=(10, 6))
    
    ax = sns.boxplot(
        data=metrics_df,
        x="horizon",
        y=metric,
        hue="model_type",      # compare models by horizon
        palette="Set3",
        showfliers=False
    )
    
    sns.stripplot(
        data=metrics_df,
        x="horizon",
        y=metric,
        hue="model_type",
        dodge=True,
        alpha=0.35,
        size=2,
        color="black",
        legend=False
    )
    
    handles, labels = ax.get_legend_handles_labels()
    # keep only first legend (from boxplot)
    if handles:
        ax.legend(handles[:len(set(metrics_df['model_type']))],
                  labels[:len(set(metrics_df['model_type']))],
                  title="Model", frameon=True)
    
    plt.title(title if title else metric.replace("_", " ").title(), fontsize=15)
    plt.xlabel("Horizon", fontsize=12)
    plt.ylabel(metric.replace("_"," ").title(), fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T20:41:06.842998Z","iopub.execute_input":"2025-11-11T20:41:06.843254Z","iopub.status.idle":"2025-11-11T21:05:01.313545Z","shell.execute_reply.started":"2025-11-11T20:41:06.843232Z","shell.execute_reply":"2025-11-11T21:05:01.312441Z"},"jupyter":{"outputs_hidden":false}}
# Assuming you have:
# - dataframes: list of 30 daily dataframes
# - feature_cols: list of your 60 feature names

# Run the dual-head pipeline
results, eng_cols = complete_training_pipeline_dual(
    dataframes=dataframes,
    base_feature_cols=feature_cols,
    train_ratio=0.8,
    lookback=18,
    price_col='Price'
)

# Access trained models
tcn_dual_model = results['tcn']['model']
gru_dual_model = results['gru']['model']
transformer_dual_model = results['transformer']['model']



# Save the models permanently
tcn_dual_model.save("/kaggle/working/tcn_1m_2m_dualhead.keras")
gru_dual_model.save("/kaggle/working/gru_3m_4m_dualhead.keras")
transformer_dual_model.save("/kaggle/working/transformer_5m_dualhead.keras")

print("Successfully saved models")

# %% [code] {"execution":{"iopub.status.busy":"2025-11-11T21:05:01.314030Z","iopub.status.idle":"2025-11-11T21:05:01.314293Z","shell.execute_reply.started":"2025-11-11T21:05:01.314173Z","shell.execute_reply":"2025-11-11T21:05:01.314185Z"},"jupyter":{"outputs_hidden":false}}

for model_type, info in results.items():
    eval_results = info['eval']
    horizons = info['horizons']
    
    for day_metrics in eval_results:
        day = day_metrics['day_idx']
        
        for h in horizons:
            entry = {
                "day": day,
                "horizon": h,
                "model_type": model_type,
                "cls_acc": day_metrics[f"h{h}_cls_acc"],
                "reg_da": day_metrics[f"h{h}_reg_da"],
                "combined_da": day_metrics[f"h{h}_combined_da"],
                "big_move_da": day_metrics.get(f"h{h}_da_big", None),   # optional
                "corr": np.nan,      # simple version
                "sharpe": np.nan     # simple version
            }
            all_metrics.append(entry)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-11T21:05:01.315003Z","iopub.status.idle":"2025-11-11T21:05:01.315262Z","shell.execute_reply.started":"2025-11-11T21:05:01.315146Z","shell.execute_reply":"2025-11-11T21:05:01.315157Z"}}
metrics_df = pd.DataFrame(all_metrics)
metrics_path = "/kaggle/working/dualhead_metrics_all_models.csv"
metrics_df.to_csv(metrics_path, index=False)

metrics_to_plot = [
    "cls_acc",
    "reg_da",
    "combined_da",
    "big_move_da",
    "corr",
    "sharpe"
]

for metric in metrics_to_plot:
    plot_metric_boxplot(
        metrics_df, 
        metric,
        f"{metric.replace('_',' ').title()} Across Horizons (All Models)"
    )