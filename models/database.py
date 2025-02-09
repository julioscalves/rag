from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base

import settings


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()

    cursor.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead logging
    cursor.execute(
        "PRAGMA synchronous=NORMAL;"
    )  # Define the synchronization between the database's data and the cache
    cursor.execute(
        "PRAGMA temp_store=MEMORY;"
    )  # Forces the temporary storage to be in RAM
    cursor.execute("PRAGMA cache_size=-64000;")  # Define RAM cache size

    cursor.close()


engine = create_engine(f"sqlite:///{settings.DB_FILENAME}")
session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()
Base.query = session.query_property()
