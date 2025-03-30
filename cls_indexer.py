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

    def __getitem__(self, item: Hashable) -> int:
        """Get the index of an item."""
        if item not in self.idxMap:
            raise KeyError(f"Item '{item}' not found in indexer.")
        return self.idxMap[item]

    def getAll(self) -> list[Hashable]:
        """Get all items."""
        return self.items