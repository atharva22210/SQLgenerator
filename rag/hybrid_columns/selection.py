"""Column descriptions, mandatory join keys and reduced schema context."""
from sqlglot import exp


def column_text(doc: dict, column: dict) -> str:
    """Describe one column for independent dense scoring, excluding labels/IDs."""
    return (f"Table {doc['table']}. Column {column['name']} {column['type']}. "
            f"{column.get('comment', '')} Domain: {doc['domain']} {doc['domain_description']}")


def required_columns(documents: list[dict]) -> dict[str, set[str]]:
    """Preserve PKs, local FK columns and selected tables' referenced columns.

    A FK may point to a unique column other than the target PK. Keep that
    target column too when its table is selected. Composite keys retain ALL
    components, even if none score well against the question.
    """
    lookup = {(d["db_id"], d["table"]): d["document_id"] for d in documents}
    required = {d["document_id"]: set(d["primary_key"]) for d in documents}
    for doc in documents:
        for key in doc["foreign_keys"]:
            required[doc["document_id"]].update(key["columns"])
            target = lookup.get((doc["db_id"], key["referenced_table"]))
            if target is not None:
                required[target].update(key["referenced_columns"])
    for doc in documents:
        if not required[doc["document_id"]] <= {c["name"] for c in doc["columns"]}:
            raise ValueError(f"Key references an unknown column in {doc['document_id']}")
    return required


def render_schema_sql(doc: dict) -> str:
    """Render only retained columns and declared PK/FKs as MySQL context DDL.

    This is a schema sketch for generation, not a migration script: unrelated
    source constraints/defaults are not reconstructed. Quote identifiers so a
    column named 'order' stays valid SQL. No SQL is executed.
    """
    def identifier(name):
        return exp.to_identifier(name, quoted=True).sql(dialect="mysql")

    def identifiers(names):
        return ", ".join(identifier(name) for name in names)

    lines = [f"{identifier(c['name'])} {c['type']}" for c in doc["columns"]]
    if doc["primary_key"]:
        lines.append(f"PRIMARY KEY ({identifiers(doc['primary_key'])})")
    for key in doc["foreign_keys"]:
        lines.append(f"FOREIGN KEY ({identifiers(key['columns'])}) REFERENCES "
                     f"{identifier(key['referenced_table'])} ({identifiers(key['referenced_columns'])})")
    return f"CREATE TABLE {identifier(doc['table'])} (\n  " + ",\n  ".join(lines) + "\n)"
