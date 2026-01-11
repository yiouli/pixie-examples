"""
Lightweight SQL Database Utilities

Drop-in replacements for langchain_community.utilities.SQLDatabase and
langchain_community.agent_toolkits.SQLDatabaseToolkit without the heavy dependencies.

Built on top of SQLAlchemy for database interaction.
"""

from typing import Any, Optional
from sqlalchemy import create_engine, inspect, text, MetaData
from sqlalchemy.engine import Engine


class SQLDatabase:
    """Lightweight SQLAlchemy database wrapper."""

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        include_tables: Optional[list[str]] = None,
        ignore_tables: Optional[list[str]] = None,
        sample_rows_in_table_info: int = 3,
    ):
        """Initialize SQLDatabase wrapper.

        Args:
            engine: SQLAlchemy engine
            schema: Optional schema name
            include_tables: Optional list of tables to include
            ignore_tables: Optional list of tables to ignore
            sample_rows_in_table_info: Number of sample rows to include in table info
        """
        self._engine = engine
        self._schema = schema
        self._include_tables = include_tables
        self._ignore_tables = ignore_tables or []
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._metadata = MetaData()

    @classmethod
    def from_uri(
        cls,
        database_uri: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> "SQLDatabase":
        """Create SQLDatabase from a database URI.

        Args:
            database_uri: Database connection URI (e.g., 'sqlite:///path/to/db.db')
            engine_args: Optional arguments to pass to create_engine
            **kwargs: Additional arguments for SQLDatabase initialization

        Returns:
            SQLDatabase instance
        """
        engine_args = engine_args or {}
        engine = create_engine(database_uri, **engine_args)
        return cls(engine, **kwargs)

    @property
    def dialect(self) -> str:
        """Get database dialect name."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> list[str]:
        """Get list of usable table names.

        Returns:
            List of table names
        """
        inspector = inspect(self._engine)
        all_tables = inspector.get_table_names(schema=self._schema)

        # Filter tables
        if self._include_tables:
            tables = [t for t in all_tables if t in self._include_tables]
        else:
            tables = [t for t in all_tables if t not in self._ignore_tables]

        return tables

    def get_table_info(self, table_names: Optional[list[str]] = None) -> str:
        """Get information about specified tables including schema and sample rows.

        Args:
            table_names: Optional list of table names. If None, uses all usable tables.

        Returns:
            Formatted string with table information
        """
        if table_names is None:
            table_names = self.get_usable_table_names()

        inspector = inspect(self._engine)
        table_info = []

        for table_name in table_names:
            # Get columns
            columns = inspector.get_columns(table_name, schema=self._schema)

            # Build CREATE TABLE statement
            create_table = f"CREATE TABLE {table_name} (\n"
            column_defs = []
            for col in columns:
                col_type = str(col["type"])
                nullable = "" if col.get("nullable", True) else " NOT NULL"
                column_defs.append(f"  {col['name']} {col_type}{nullable}")
            create_table += ",\n".join(column_defs)
            create_table += "\n)"

            table_info.append(create_table)

            # Get sample rows
            if self._sample_rows_in_table_info > 0:
                try:
                    query = text(
                        f"SELECT * FROM {table_name} LIMIT {self._sample_rows_in_table_info}"
                    )
                    with self._engine.connect() as conn:
                        result = conn.execute(query)
                        rows = result.fetchall()
                        if rows:
                            table_info.append(
                                f"\n{self._sample_rows_in_table_info} rows from {table_name} table:"
                            )
                            for row in rows:
                                table_info.append(str(row))
                except Exception as e:
                    table_info.append(f"\nError fetching sample rows: {e}")

            table_info.append("")  # Empty line between tables

        return "\n".join(table_info)

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return results.

        Args:
            command: SQL command to execute
            fetch: How to fetch results ('all', 'one', or 'cursor')

        Returns:
            Query results as a string
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(command))

            if fetch == "all":
                rows = result.fetchall()
                return str(rows)
            elif fetch == "one":
                row = result.fetchone()
                return str(row)
            else:
                return str(list(result))


class SQLDatabaseToolkit:
    """Toolkit for creating LangChain tools to interact with SQL databases."""

    def __init__(self, db: SQLDatabase, llm: Any):
        """Initialize SQL Database Toolkit.

        Args:
            db: SQLDatabase instance
            llm: Language model (not used in tool execution, but kept for compatibility)
        """
        self._db = db
        self._llm = llm

    def get_tools(self) -> list:
        """Get list of tools for SQL database interaction.

        Returns:
            List of LangChain Tool objects
        """
        from langchain_core.tools import StructuredTool

        tools = [
            StructuredTool.from_function(
                name="sql_db_list_tables",
                description="List all available tables in the database. "
                "Use this to see what tables you can query. "
                "Input should be an empty string.",
                func=self._list_tables,
            ),
            StructuredTool.from_function(
                name="sql_db_schema",
                description="Get the schema and sample rows for specified tables. "
                "Input should be a comma-separated list of table names. "
                "Example input: 'table1, table2, table3'",
                func=self._get_schema,
            ),
            StructuredTool.from_function(
                name="sql_db_query",
                description="Execute a SQL query against the database. "
                "Input should be a valid SQL query. "
                "Returns the query results. "
                "If the query is not correct, an error message will be returned. "
                "If an error is returned, rewrite the query and try again.",
                func=self._query,
            ),
        ]

        return tools

    def _list_tables(self, _: str = "") -> str:
        """List all tables in the database."""
        tables = self._db.get_usable_table_names()
        return ", ".join(tables)

    def _get_schema(self, table_names: str) -> str:
        """Get schema for specified tables.

        Args:
            table_names: Comma-separated list of table names

        Returns:
            Formatted schema information
        """
        # Parse table names
        tables = [t.strip() for t in table_names.split(",")]
        return self._db.get_table_info(tables)

    def _query(self, query: str) -> str:
        """Execute SQL query.

        Args:
            query: SQL query to execute

        Returns:
            Query results or error message
        """
        try:
            result = self._db.run(query)
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"
