from typing import Hashable, Sequence


class Indexer:
    """
    A class to look for the index of an item in multiple lists.

    Attributes:
        items (list): A list of items to be indexed.
        idxMap (dict): A dictionary mapping items to their indices.
    """

    def __init__(self, *items: Sequence[Hashable]) -> None:
        """
        Initialize the Indexer with a list of items.

        Args:
            items (list): A list of items to be indexed.
        """

        self.items = []
        for item in items:
            if isinstance(item, list):
                self.items.extend(item)
            else:
                self.items.append(item)

        self.idxMap = {item: i for i, item in enumerate(self.items)}

    def getItem(self, item: Hashable) -> int:
        """Get the index of an item."""
        if item not in self.idxMap:
            raise KeyError(f"Item '{item}' not found in indexer.")
        return self.idxMap[item]

    def getItems(self, items: Sequence[Hashable], sort=False) -> list[int]:
        """Get the indices of a list of items."""
        out = [self[item] for item in items]
        if sort:
            out.sort()
        return out

    def slicerOf(self, items: Sequence[Hashable], dim=0):
        slices = [slice(None)] * dim + self.getItems(items)
        return tuple(slices)

    def getAll(self) -> list[Hashable]:
        """Get all items."""
        return self.items

    def __getitem__(self, item: Hashable) -> int:
        return self.getItem(item)
