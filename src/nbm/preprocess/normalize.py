import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler


# create the angle encoder
def encodeAngles(X):
    """Convert angles to sin and cos components"""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    radians = np.radians(X)
    sins = np.sin(radians)
    coss = np.cos(radians)
    # push the values to the range of [0, 1]
    sins = (sins + 1) / 2
    coss = (coss + 1) / 2
    return np.column_stack([sins, coss])


def decodeAngles(X):
    """Convert sin and cos components back to angles"""
    X = np.asarray(X)
    n_features = X.shape[1] // 2

    sins = X[:, :n_features]
    coss = X[:, n_features:]

    # [0, 1] → [-1, 1]
    sins = 2 * sins - 1
    coss = 2 * coss - 1

    angles = np.degrees(np.arctan2(sins, coss)) % 360

    return angles


def encodedAngleNames(_theTransformerObj, featNames):
    """Generate names for the encoded angle features"""
    return [f"{feat}_sin" for feat in featNames] + [f"{feat}_cos" for feat in featNames]


def createFeatureTransformer(rangedFeatRanges, angleFeats, immuteFeats):
    """Create a feature transformer for target, angle, and immutable features.

    The transformer will:
    - Scale target features to a range of [0, 1]
    - Encode angle features as sine and cosine components
    - Pass through immutable features unchanged

    Parameters
    ----------
    rangedFeatRanges : dict[str, tuple[float, float]]
        Dictionary mapping feature names to their value ranges (min, max)
    angleFeats : list
        List of angle features to be encoded as sine and cosine components
    immuteFeats : list
        List of immutable features that will not be transformed

    Returns
    -------
    ColumnTransformer
        Configured ColumnTransformer that applies the specified transformations.
        Composed of: "range", "angle", and "immute" transformers.

    Examples
    --------
    >>> transformer = create_feature_transformer(
    ...     targetFeats=['age', 'income'],
    ...     angleFeats=['orientation'],
    ...     immuteFeats=['id', 'timestamp']
    ... )
    """

    rangedFeats = list(rangedFeatRanges.keys())

    # the following features need special treatment
    ## angle features
    specialFeats = angleFeats + []  # reserved for other special features

    # scaler based on the usual value ranges
    rangeScaler = MinMaxScaler()
    rangeData = np.array(
        [
            [rangedFeatRanges[feat][0] for feat in rangedFeats],  # min values
            [rangedFeatRanges[feat][1] for feat in rangedFeats],  # max values
        ]
    )

    angleEncoder = FunctionTransformer(
        encodeAngles,
        decodeAngles,
        feature_names_out=encodedAngleNames,
        check_inverse=False,
    )

    # stack the transformers by column
    ## identify the indices of the features
    allFeats = rangedFeats + specialFeats + immuteFeats
    regularFeatIndices = [allFeats.index(feat) for feat in rangedFeats]
    angleFeatIndices = [allFeats.index(feat) for feat in angleFeats]
    immuteFeatIndices = [allFeats.index(feat) for feat in immuteFeats]
    transformer = ColumnTransformer(
        transformers=[
            ("range", rangeScaler, regularFeatIndices),
            ("angle", angleEncoder, angleFeatIndices),
            ("immute", "passthrough", immuteFeatIndices),
        ],
        verbose_feature_names_out=False,
        remainder="drop",
    )

    # fit on dummy data
    dummyData = np.random.rand(
        10, len(regularFeatIndices) + len(angleFeatIndices) + len(immuteFeatIndices)
    )
    transformer.fit(dummyData)

    # fit the range scaler
    rangeScaler = transformer.named_transformers_["range"]
    rangeScaler.fit(rangeData)

    return transformer
