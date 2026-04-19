import asyncio
from app.db.session import engine, async_session
from app.db.models import SequenceRecord
from sqlalchemy import select

async def main():
    async with engine.begin() as conn:
        pass
    
    async with async_session() as session:
        try:
            r = SequenceRecord(sequence_data='ATGC', length=4, gc_content=50.0, is_valid=1)
            session.add(r)
            await session.commit()
            print('Success')
        except Exception as e:
            print('ERROR:', e)

asyncio.run(main())
