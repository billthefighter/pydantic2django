# XML Schema Notes: complexType Handling

This note documents the current level of `xs:complexType` support in the XML Schema parser and outlines next steps for improved fidelity.

## What’s supported today

- Content models inside `xs:complexType`:
  - `xs:sequence`, `xs:choice`, `xs:all` are parsed and their child `xs:element` entries are collected onto the type.
  - `xs:attribute` entries on the type are parsed and attached.
- `xs:simpleContent` (minimal):
  - We capture attributes declared on the `xs:extension` node so attribute-only types aren’t dropped.
  - Other `simpleContent` patterns remain limited and may emit warnings.
- `xs:complexContent` (minimal but usable):
  - `xs:extension`: the derived type inherits elements and attributes from the base complex type, then parses any locally declared particles (`sequence`/`choice`/`all`) and attributes. This enables basic inheritance in common schemas.
  - `xs:restriction`: currently treated like an extension (inherits base members); local particles/attributes are parsed. Restriction facets are not enforced yet. A warning is logged to indicate limited handling.

Notes and limits:
- Base type lookup assumes the same schema target namespace. Cross-schema/import resolution is limited.
- `xs:group`, `xs:attributeGroup`, `xs:any`, and `xs:anyAttribute` are not yet handled.
- Restriction semantics (e.g., occurrence tightening, element/attribute prohibition) are not enforced.
- Mixed content and substitution groups are not modeled.
- Identity constraints (`xs:key`/`xs:keyref`) are parsed at the schema level but not enforced during ingest yet.

## Impact on ingestion

- Derived complex types appear with inherited elements/attributes, so ingestion can proceed for most extension-based hierarchies.
- Because restriction rules aren’t enforced, ingest may accept instances that a strict validator would reject. If strict validation is required, keep producer-side validation enabled or post-validate with a dedicated XML library.

## Recommended next steps

1. Complex content completeness
   - Implement `xs:group` and `xs:attributeGroup` expansion.
   - Add support for `xs:any` / `xs:anyAttribute` (map to flexible fields, e.g., JSON or key-value side tables; make this configurable).
   - Resolve base types across imported schemas and different namespaces.
2. Proper restriction handling
   - Enforce occurrence constraints and member prohibitions for `xs:restriction`.
   - Apply facet-driven narrowing where applicable.
3. Mixed content and substitution groups
   - Add mixed content support (e.g., serialize as text + children, or a structured representation).
   - Implement substitution group resolution for element polymorphism.
4. Identity constraints
   - Add a post-pass to resolve `xs:key` / `xs:keyref` (e.g., ID/IDREF) during ingestion.
5. Diagnostics & configuration
   - Reduce warnings once features are implemented; add a strict mode to fail on unsupported constructs.
   - Expose per-feature toggles (e.g., treat restriction as error vs. extension-like fallback).

## Practical guidance

- If you encounter warnings about unsupported `complexContent` but your schema mostly uses `xs:extension`, the current implementation will usually be sufficient for ingesting instances.
- For schemas relying heavily on `xs:restriction`, consider flattening types or relaxing validation on the ingest path until restriction semantics are implemented.
- You can silence specific parser warnings by adjusting the logger for `pydantic2django.xmlschema.parser` if the limits don’t affect your ingestion semantics.

## References

- Implementation locations:
  - Parser: `src/pydantic2django/xmlschema/parser.py`
  - Ingestor: `src/pydantic2django/xmlschema/ingestor.py`
