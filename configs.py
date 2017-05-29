import collections

Config = collections.namedtuple(
    'Config',
    [
        'size',
        'alphabet_size',
        'num_layers',
        'batch_size',
        'num_steps',
        'dropout',
        'cell_kind',
        'learning_rate',
    ])


def get_configs(alphabet_size):
    OneLayerD0 = Config(
        size=200,
        alphabet_size=alphabet_size,
        num_layers=1,
        batch_size=20,
        num_steps=20,
        dropout=0.0,
        cell_kind=True,
        learning_rate=1.0
    )

    OneLayerD2 = Config(
        size=200,
        alphabet_size=alphabet_size,
        num_layers=1,
        batch_size=20,
        num_steps=20,
        dropout=0.2,
        cell_kind=True,
        learning_rate=1.0
    )

    DoubleLayerD0 = Config(
        size=200,
        alphabet_size=alphabet_size,
        num_layers=2,
        batch_size=20,
        num_steps=20,
        dropout=0.0,
        cell_kind=True,
        learning_rate=1.0
    )

    DoubleLayerD2 = Config(
        size=200,
        alphabet_size=alphabet_size,
        num_layers=2,
        batch_size=20,
        num_steps=20,
        dropout=0.2,
        cell_kind=True,
        learning_rate=1.0
    )

    TripleLayerD2 = Config(
        size=200,
        alphabet_size=alphabet_size,
        num_layers=3,
        batch_size=20,
        num_steps=20,
        dropout=0.2,
        cell_kind=True,
        learning_rate=1.0
    )

    NASOneLayerD2 = Config(
        size=200,
        alphabet_size=alphabet_size,
        num_layers=1,
        batch_size=20,
        num_steps=20,
        dropout=0.2,
        cell_kind=False,
        learning_rate=1.0
    )

    NASDoubleLayerD2 = Config(
        size=200,
        alphabet_size=alphabet_size,
        num_layers=2,
        batch_size=20,
        num_steps=20,
        dropout=0.2,
        cell_kind=False,
        learning_rate=1.0
    )


    return [
        ('OneLayerD0', OneLayerD0),
        ('OneLayerD2', OneLayerD2),
        # ('DoubleLayerD0', DoubleLayerD0),
        ('DoubleLayerD2', DoubleLayerD2),
        # ('TripleLayerD2', TripleLayerD2),
        ('NASOneLayerD2', NASOneLayerD2),
        # ('NASDoubleLayerD2', NASDoubleLayerD2)
        ]
