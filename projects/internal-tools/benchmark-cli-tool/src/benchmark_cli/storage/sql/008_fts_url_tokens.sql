-- Index URL hosts and paths as words. The english parser otherwise keeps a URL
-- as one url/url_path token, so a question for "portal" cannot match
-- https://host/bmtechllc/portal/ inside chunk text or metadata.
DO $$
DECLARE
    expression text;
BEGIN
    SELECT pg_get_expr(ad.adbin, ad.adrelid)
      INTO expression
    FROM pg_attrdef ad
    JOIN pg_attribute attribute
      ON attribute.attrelid = ad.adrelid
     AND attribute.attnum = ad.adnum
    JOIN pg_class class
      ON class.oid = ad.adrelid
    JOIN pg_namespace namespace
      ON namespace.oid = class.relnamespace
    WHERE class.relname = 'document_chunks'
      AND attribute.attname = 'search_vector'
      AND namespace.nspname = current_schema();

    IF expression IS NOT NULL AND position('regexp_replace' IN expression) > 0 THEN
        RETURN;
    END IF;

    EXECUTE 'DROP INDEX IF EXISTS document_chunks_search_vector_gin';
    EXECUTE 'ALTER TABLE document_chunks DROP COLUMN IF EXISTS search_vector';
    EXECUTE $ddl$
        ALTER TABLE document_chunks
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english', chunk_text) ||
                to_tsvector(
                    'english',
                    regexp_replace(chunk_text, '[^[:alnum:]]+', ' ', 'g')
                ) ||
                to_tsvector(
                    'english',
                    regexp_replace(coalesce(metadata ->> 'url', ''), '[^[:alnum:]]+', ' ', 'g')
                )
            ) STORED
    $ddl$;
    EXECUTE 'CREATE INDEX document_chunks_search_vector_gin ON document_chunks USING gin (search_vector)';
END $$;
