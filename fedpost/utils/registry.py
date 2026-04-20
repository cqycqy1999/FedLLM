from __future__ import annotations


class Registry:
    _store: dict[str, dict[str, object]] = {}

    @classmethod
    def register(cls, category: str, name: str):
        def wrapper(obj):
            cls._store.setdefault(category, {})
            cls._store[category][name] = obj
            return obj
        return wrapper

    @classmethod
    def get(cls, category: str, name: str):
        if category not in cls._store or name not in cls._store[category]:
            raise KeyError(f"{name} is not registered under category {category}")
        return cls._store[category][name]
