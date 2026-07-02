from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ITEM_COLUMNS = {
    "item_id",
    "title",
    "body",
    "topic",
    "author_id",
    "publish_time",
    "quality_score",
}
USER_COLUMNS = {"user_id"}
INTERACTION_COLUMNS = {
    "user_id",
    "item_id",
    "event_type",
    "event_time",
    "dwell_time",
}
VALID_EVENT_TYPES = {"click", "like", "collect", "skip"}


class ContentItem(BaseModel):
    item_id: str
    title: str
    body: str
    topic: str
    author_id: str
    publish_time: datetime
    quality_score: float = Field(ge=0.0, le=1.0)


class UserProfile(BaseModel):
    user_id: str
    signup_time: datetime | None = None
    segment: str | None = None


class UserInteraction(BaseModel):
    user_id: str
    item_id: str
    event_type: Literal["click", "like", "collect", "skip"]
    event_time: datetime
    dwell_time: float = Field(ge=0.0)
