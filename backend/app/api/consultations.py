from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.auth.deps import CurrentUser, SessionDep
from app.models import Consultation
from app.schemas.consultation import (
    ConsultationCreate,
    ConsultationListItem,
    ConsultationRead,
    ConsultationUpdate,
    ranked_bikes,
)

router = APIRouter(prefix="/api/consultations", tags=["consultations"])


async def get_owned_consultation(session: SessionDep, consultation_id: int, user_id: int) -> Consultation:
    consultation = await session.scalar(
        select(Consultation).where(Consultation.id == consultation_id).options(selectinload(Consultation.messages))
    )
    # 404 statt 403, um die Existenz fremder Beratungen nicht preiszugeben
    if consultation is None or consultation.owner_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Beratung nicht gefunden")
    return consultation


@router.get("", response_model=list[ConsultationListItem])
async def list_consultations(user: CurrentUser, session: SessionDep) -> list[ConsultationListItem]:
    query = select(Consultation).where(Consultation.owner_id == user.id).order_by(Consultation.updated_at.desc())
    return [
        ConsultationListItem.model_validate(c).model_copy(update={"bikes": ranked_bikes(c.recommendation)})
        for c in await session.scalars(query)
    ]


@router.post("", response_model=ConsultationRead, status_code=status.HTTP_201_CREATED)
async def create_consultation(data: ConsultationCreate, user: CurrentUser, session: SessionDep) -> Consultation:
    consultation = Consultation(owner_id=user.id, title=data.title, agent_history=[])
    session.add(consultation)
    await session.commit()
    return await get_owned_consultation(session, consultation.id, user.id)


@router.get("/{consultation_id}", response_model=ConsultationRead)
async def read_consultation(consultation_id: int, user: CurrentUser, session: SessionDep) -> Consultation:
    return await get_owned_consultation(session, consultation_id, user.id)


@router.patch("/{consultation_id}", response_model=ConsultationRead)
async def update_consultation(
    consultation_id: int, data: ConsultationUpdate, user: CurrentUser, session: SessionDep
) -> Consultation:
    consultation = await get_owned_consultation(session, consultation_id, user.id)
    if data.title is not None:
        consultation.title = data.title
    await session.commit()
    return await get_owned_consultation(session, consultation_id, user.id)


@router.delete("/{consultation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_consultation(consultation_id: int, user: CurrentUser, session: SessionDep) -> None:
    consultation = await get_owned_consultation(session, consultation_id, user.id)
    await session.delete(consultation)
    await session.commit()
