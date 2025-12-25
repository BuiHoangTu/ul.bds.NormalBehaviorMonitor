import pickle
from typing import Hashable, Sequence

import numpy as np


def sliceByTag(
    array, dim: int, tags: Sequence[Hashable], mapping
):
    # check array not empty
    if len(array) == 0:
        return array
    
    indices = [mapping[tag] for tag in tags]
    slices = [slice(None)] * dim + [indices]

    return array[tuple(slices)]


class Indexer:
    """
    A class to look for the index of an item in multiple lists.

    Attributes:
        tags (list): A list of tags to be indexed.
        idxMap (dict): A dictionary mapping items to their indices.
    """

    def __init__(
        self,
        defaultDim=0,
        *tags: Sequence[Hashable],
        **groups: Sequence[Hashable],
    ) -> None:
        """
        Initialize the Indexer with a list of items.

        Args:
            items (list): A list of items to be indexed.
        """
        self.defaultDim = defaultDim

        tagsMerged = []

        for tag in tags:
            if isinstance(tag, list):
                tagsMerged.extend(tag)
            else:
                tagsMerged.append(tag)

        for _, tag in groups.items():
            tagsMerged.extend(tag)

        self.idxMap = {tag: i for i, tag in enumerate(tagsMerged)}
        self.groups = groups

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def getId(self, tag: Hashable) -> int:
        """Get the index of an tag."""
        if tag not in self.idxMap:
            raise KeyError(f"Tag '{tag}' not found in indexer.")
        return self.idxMap[tag]

    def getIdx(self, tags: Sequence[Hashable], sort=False) -> list[int]:
        """Get the indices of a list of tags."""
        out = [self[tag] for tag in tags]
        if sort:
            out.sort()
        return out

    def slice(self, array, tags: Sequence[Hashable], dim=None):
        return sliceByTag(array, dim or self.defaultDim, tags, self.idxMap)

    def getAllTags(self) -> list[Hashable]:
        """Get all tags."""
        return list(self.idxMap.keys())

    def getGroups(self) -> list[str]:
        return list(self.groups.keys())

    def getGroupTags(self, group: str) -> Sequence[Hashable]:
        if group not in self.groups:
            raise KeyError(f"Group '{group}' not found in indexer.")
        return self.groups[group]

    def getIdxInGroup(self, group: str) -> Sequence[int]:
        return self.getIdx(self.getGroupTags(group))

    def __len__(self) -> int:
        return len(self.idxMap)

    def __contains__(self, tag: Hashable) -> bool:
        return tag in self.idxMap

    def __getitem__(self, tag: Hashable) -> int:
        return self.getId(tag)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Indexer):
            return False

        return (
            self.defaultDim == value.defaultDim
            and self.groups == value.groups
            and self.idxMap == value.idxMap
        )
