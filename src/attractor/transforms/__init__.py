from attractor.transforms.variable_expansion import VariableExpansionTransform
from attractor.transforms.stylesheet import StylesheetApplicationTransform

BUILTIN_TRANSFORMS = [
    VariableExpansionTransform(),
    StylesheetApplicationTransform(),
]


def apply_transforms(graph, custom_transforms=None):
    """Apply all built-in transforms (and any custom ones) to *graph*."""
    transforms = list(BUILTIN_TRANSFORMS)
    if custom_transforms:
        transforms.extend(custom_transforms)
    for t in transforms:
        graph = t.apply(graph)
    return graph
