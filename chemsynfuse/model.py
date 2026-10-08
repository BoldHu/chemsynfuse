"""ChemSynFuse predictor over externally prepared feature tensors."""
from dataclasses import dataclass
from .fusion import HierarchicalFusion
from .adaptation import FusionPredictor


@dataclass
class ModelSchema:
    """Dimensions only; contains no dataset or preprocessing implementation."""
    fields: int
    categories: int
    metadata_dim: int
    summary_dim: int

    def __post_init__(self):
        if min(self.fields, self.categories) < 1 or min(self.metadata_dim, self.summary_dim) < 0:
            raise ValueError('Invalid input dimensions')

    @property
    def state(self):
        return {'fields': range(self.fields), 'categories': range(self.categories)}


class ChemSynFuse(FusionPredictor):
    """Shared fusion backbone with a classification or scalar regression head.

    Molecular features are obtained with MolecularPretrainer.encode using a
    frozen, externally supplied pretrained encoder, as in downstream training.
    """
    def __init__(self, schema, kind='classification', classes=4, molecule_dim=256, dim=64):
        if kind not in ('classification', 'regression'):
            raise ValueError('Unknown prediction task')
        if kind == 'classification' and classes < 2:
            raise ValueError('At least two classes are required')
        super().__init__(HierarchicalFusion(schema, molecule_dim, dim), classes if kind == 'classification' else 1)
        self.kind = kind
        self.classes = classes
