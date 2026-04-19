import asyncio
from app.db.session import engine, Base
from app.db.models import SequenceRecord, Experiment, MutationLog, SharedResult

async def main():
    async with engine.begin() as conn:
        print('Creating tables...')
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        print('Done.')

asyncio.run(main())
