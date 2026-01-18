import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None
    id: str | None = None  # Qdrant point ID (UUID)
    score: float | None = None  # Similarity score for sorting multi-collection results


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None) -> str:
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        :return: The point ID (UUID) of the stored entry.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        point_id = uuid.uuid4().hex
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )
        return point_id

    async def retrieve(
        self, point_id: str, *, collection_name: str | None = None
    ) -> Entry | None:
        """
        Retrieve a point by exact ID from the Qdrant collection.
        :param point_id: The exact point ID (UUID) to retrieve.
        :param collection_name: The name of the collection to retrieve from, optional. If not provided,
                                the default collection is used.
        :return: The entry if found, None otherwise.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        try:
            # Try to parse as integer first (for collections with integer IDs)
            try:
                parsed_id = int(point_id)
            except ValueError:
                parsed_id = point_id  # Keep as string (UUID)

            result = await self._client.retrieve(
                collection_name=collection_name,
                ids=[parsed_id],
                with_payload=True,
                with_vectors=False,
            )

            if not result:
                return None

            point = result[0]
            return Entry(
                id=str(point.id),
                content=point.payload.get("page_content") or point.payload.get("document", ""),
                metadata=point.payload.get(METADATA_PATH, {}),
            )
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            Entry(
                content=result.payload.get("page_content")
                or result.payload.get("document", ""),
                metadata=result.payload.get("metadata"),
                id=str(result.id),  # Capture point ID
                score=result.score,  # Capture similarity score
            )
            for result in search_results.points
        ]

    async def search_multi(
        self,
        query: str,
        *,
        collections: list[str] | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        limit_multiplier: int = 3,
    ) -> list[Entry]:
        """
        Search multiple collections in parallel and merge results.

        :param query: The query to use for the search.
        :param collections: List of collection names to search, or None for default, or ["*"] for all collections.
        :param limit: The maximum number of entries to return per collection.
        :param query_filter: The filter to apply to the query, if any.
        :param limit_multiplier: Multiplier for total results when searching multiple collections (default 3x).
        :return: A list of entries found, sorted by score.
        """
        # Determine which collections to search
        collections_to_search: list[str]
        if collections is None:
            if self._default_collection_name is None:
                return []
            collections_to_search = [self._default_collection_name]
        elif collections == ["*"]:
            collections_to_search = await self.get_collection_names()
        else:
            collections_to_search = collections

        # Embed query once (reuse for all collections)
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search all collections in parallel
        import asyncio

        search_tasks = [
            self._search_single_collection(
                collection, query_vector, vector_name, limit, query_filter
            )
            for collection in collections_to_search
        ]

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge results, filter errors, add collection metadata
        all_entries: list[Entry] = []
        for collection, result in zip(collections_to_search, results):
            if isinstance(result, BaseException):
                logger.warning(f"Collection {collection} search failed: {result}")
                continue

            # Add collection name to metadata
            entries: list[Entry] = result
            for entry in entries:
                if entry.metadata is None:
                    entry.metadata = {}
                entry.metadata["collection"] = collection
                all_entries.append(entry)

        # Sort by score (descending)
        all_entries.sort(
            key=lambda e: e.score if e.score is not None else 0.0, reverse=True
        )

        # Return top N (smart limiting for multi-collection)
        total_limit = (
            limit * min(len(collections_to_search), limit_multiplier)
            if len(collections_to_search) > 1
            else limit
        )
        return all_entries[:total_limit]

    async def _search_single_collection(
        self,
        collection: str,
        query_vector: list[float],
        vector_name: str,
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[Entry]:
        """
        Helper to search a single collection.

        :param collection: The name of the collection to search.
        :param query_vector: The pre-embedded query vector.
        :param vector_name: The name of the vector field.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.
        :return: A list of entries found.
        """
        if not await self._client.collection_exists(collection):
            logger.warning(f"Collection {collection} does not exist, skipping")
            return []

        search_results = await self._client.query_points(
            collection_name=collection,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            Entry(
                content=result.payload.get("page_content")
                or result.payload.get("document", ""),
                metadata=result.payload.get("metadata"),
                id=str(result.id),
                score=result.score,
            )
            for result in search_results.points
        ]

    async def delete(
        self,
        filter_condition: models.Filter | None = None,
        *,
        point_ids: list[str] | None = None,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete points from the Qdrant collection by IDs or filter conditions.
        :param filter_condition: Qdrant filter to identify points to delete (optional)
        :param point_ids: List of point IDs (UUIDs) to delete (optional)
        :param collection_name: The name of the collection to delete from, optional. If not provided,
                                the default collection is used.
        :return: Dictionary with status and operation_id
        :raises ValueError: If neither filter_condition nor point_ids is provided, or if both are provided
        """
        # Validate that exactly one deletion method is provided
        if filter_condition is None and point_ids is None:
            raise ValueError("Either filter_condition or point_ids must be provided")
        if filter_condition is not None and point_ids is not None:
            raise ValueError("Cannot provide both filter_condition and point_ids")

        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            logger.warning(
                f"Collection {collection_name} does not exist, nothing to delete"
            )
            return {"status": "ok", "deleted_count": 0}

        # Choose deletion method based on parameters
        if point_ids is not None:
            # Delete by IDs (recommended - more reliable)
            points_selector = models.PointIdsList(points=point_ids)
            logger.info(
                f"Deleting {len(point_ids)} points by ID from {collection_name}"
            )
        else:
            # Delete by filter
            points_selector = models.FilterSelector(filter=filter_condition)
            logger.info(f"Deleting points by filter from {collection_name}")

        result = await self._client.delete(
            collection_name=collection_name,
            points_selector=points_selector,
        )

        logger.info(
            f"Delete operation completed for collection {collection_name}: {result}"
        )
        return {"status": result.status, "operation_id": result.operation_id}

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
