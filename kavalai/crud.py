"""
CRUD database operations that work with both backoffice and agents database.
"""

from typing import Type, TypeVar, Sequence, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase


T = TypeVar("T", bound="DeclarativeBase")


async def get_all(db: AsyncSession, model: Type[T]) -> Sequence[T]:
    """Fetch all records for the model."""
    result = await db.execute(select(model))
    return result.scalars().all()


async def get_one(db: AsyncSession, model: Type[T], id: Any) -> T | None:
    """Fetch a single record by its primary key."""
    return await db.get(model, id)


async def insert(db: AsyncSession, model: Type[T], data: dict) -> T:
    """Create a new record."""
    instance = model(**data)
    db.add(instance)
    await db.commit()
    await db.refresh(instance)
    return instance


async def update(db: AsyncSession, model: Type[T], id: Any, data: dict) -> T | None:
    """Update an existing record by ID."""
    instance = await get_one(db, model, id)
    if instance:
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        await db.commit()
        await db.refresh(instance)
    return instance


async def delete(db: AsyncSession, model: Type[T], id: Any) -> bool:
    """Delete a record by ID."""
    instance = await get_one(db, model, id)
    if instance:
        await db.delete(instance)
        await db.commit()
        return True
    return False
