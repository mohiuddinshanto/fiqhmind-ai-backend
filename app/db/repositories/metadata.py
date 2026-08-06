from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.db.models import MetadataDocument, MetadataField, MetadataStructure, PageMetadata
from app.db.repositories.base import RepositoryBase

if TYPE_CHECKING:
    from app.services.metadata import DocumentMetadata


class MetadataRepository(RepositoryBase[MetadataDocument]):
    model = MetadataDocument

    def get_by_job(self, job_id: str) -> MetadataDocument | None:
        return self._session.scalar(
            select(MetadataDocument).where(MetadataDocument.job_id == job_id)
        )

    def create_document(
        self,
        *,
        job_id: str,
        upload_id: str,
        original_filename: str,
        page_count: int,
        numbering_system: str,
        confidence: float,
    ) -> MetadataDocument:
        document = MetadataDocument(
            job_id=job_id,
            upload_id=upload_id,
            original_filename=original_filename,
            page_count=page_count,
            numbering_system=numbering_system,
            confidence=confidence,
        )
        self._session.add(document)
        self._session.flush()
        return document

    def add_field(
        self,
        document: MetadataDocument,
        *,
        field: str,
        value: str | None,
        confidence: float,
        source: str,
        details: dict | None = None,
    ) -> MetadataField:
        row = MetadataField(
            document_id=document.id,
            field=field,
            value=value,
            confidence=confidence,
            source=source,
            details=details,
        )
        self._session.add(row)
        return row

    def add_page(
        self,
        document: MetadataDocument,
        *,
        pdf_page: int,
        printed_page: str,
        printed_page_numeric: int | None,
        numbering_system: str,
        page_number_uncertain: bool,
        confidence: float,
        source: str,
        kitab: str | None = None,
        bab: str | None = None,
        fasl: str | None = None,
    ) -> PageMetadata:
        row = PageMetadata(
            document_id=document.id,
            pdf_page=pdf_page,
            printed_page=printed_page,
            printed_page_numeric=printed_page_numeric,
            numbering_system=numbering_system,
            page_number_uncertain=page_number_uncertain,
            confidence=confidence,
            source=source,
            kitab=kitab,
            bab=bab,
            fasl=fasl,
        )
        self._session.add(row)
        return row

    def add_structure(
        self,
        document: MetadataDocument,
        *,
        level: str,
        name: str,
        page_start: int,
        page_end: int | None,
        confidence: float,
        source: str,
    ) -> MetadataStructure:
        row = MetadataStructure(
            document_id=document.id,
            level=level,
            name=name,
            page_start=page_start,
            page_end=page_end,
            confidence=confidence,
            source=source,
        )
        self._session.add(row)
        return row

    def list_pages(self, document: MetadataDocument) -> list[PageMetadata]:
        return list(
            self._session.scalars(
                select(PageMetadata)
                .where(PageMetadata.document_id == document.id)
                .order_by(PageMetadata.pdf_page)
            )
        )

    def list_fields(self, document: MetadataDocument) -> list[MetadataField]:
        return list(
            self._session.scalars(
                select(MetadataField)
                .where(MetadataField.document_id == document.id)
                .order_by(MetadataField.field)
            )
        )

    def list_structures(self, document: MetadataDocument) -> list[MetadataStructure]:
        return list(
            self._session.scalars(
                select(MetadataStructure)
                .where(MetadataStructure.document_id == document.id)
                .order_by(MetadataStructure.level, MetadataStructure.page_start)
            )
        )

    def delete_for_job(self, job_id: str) -> None:
        """Remove the metadata document (and its fields/pages/structures) for a job."""
        document = self.get_by_job(job_id)
        if document is not None:
            self._session.delete(document)
            self._session.commit()

    def save_document(
        self,
        *,
        job_id: str,
        upload_id: str,
        original_filename: str,
        document: DocumentMetadata,
    ) -> MetadataDocument:
        """Persist an engine result, replacing any previous run for the job."""
        self.delete_for_job(job_id)
        row = self.create_document(
            job_id=job_id,
            upload_id=upload_id,
            original_filename=original_filename,
            page_count=document.page_count,
            numbering_system=document.numbering_system,
            confidence=document.confidence,
        )
        for field_item in document.fields:
            self.add_field(
                row,
                field=field_item.field,
                value=field_item.value,
                confidence=field_item.confidence,
                source=field_item.source,
                details=field_item.details,
            )
        for page_item in document.pages:
            self.add_page(
                row,
                pdf_page=page_item.pdf_page,
                printed_page=page_item.printed_page,
                printed_page_numeric=page_item.printed_page_numeric,
                numbering_system=page_item.numbering_system,
                page_number_uncertain=page_item.page_number_uncertain,
                confidence=page_item.confidence,
                source=page_item.source,
                kitab=page_item.kitab,
                bab=page_item.bab,
                fasl=page_item.fasl,
            )
        for structure_item in document.structures:
            self.add_structure(
                row,
                level=structure_item.level,
                name=structure_item.name,
                page_start=structure_item.page_start,
                page_end=structure_item.page_end,
                confidence=structure_item.confidence,
                source=structure_item.source,
            )
        return row
