# storage package
from storage.db import init_db
from storage.vector_db import init_collections


def init_all():
    """Initialise both storage backends."""
    init_db()
    init_collections()
