"""Common feature definitions and preprocessing utilities."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_feature_lists():
    """Return lists of numeric and categorical feature names."""
    numeric_feats = [
        'grid_position', 'Q1_sec', 'Q2_sec', 'Q3_sec',
        'month', 'avg_finish_pos', 'avg_grid_pos', 'avg_const_finish',
        'finish_rate_prev5',
        'team_qual_gap',
        'driver_points_prev', 'driver_rank_prev',
        'constructor_points_prev', 'constructor_rank_prev',

        'num_pitstops',
        'avg_pitstop_duration',
        'tyre_degradation_rate',
        'qual_delta',
        'circuit_top3_freq',
        'head_to_head_vs_teammate',

        'grid_diff', 'Q3_diff',

        # Overtakes features
        'weighted_overtakes',
        'overtakes_per_lap',
        'weighted_overtakes_per_lap',
        'ewma_overtakes_per_lap',
        'ewma_weighted_overtakes_per_lap',
    ]
    categorical_feats = ['circuit_country', 'circuit_city']
    return numeric_feats, categorical_feats


def build_preprocessor():
    """Create a ColumnTransformer for model preprocessing."""
    numeric_feats, categorical_feats = get_feature_lists()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats),
    ])
    return preprocessor
