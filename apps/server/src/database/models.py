import enum
from enum import Enum
from tortoise import fields, models

class TimestampMixin(models.Model):
  created_at = fields.DatetimeField(auto_now_add=True)
  updated_at = fields.DatetimeField(auto_now=True)
  deleted_at = fields.DatetimeField(null=True)

class Domain(TimestampMixin):
  id = fields.IntField(pk=True)
  name = fields.CharField(max_length=50)


class Capability(TimestampMixin):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    description = fields.CharField(max_length=255)
    domain = fields.ForeignKeyField('models.Domain', related_name='capabilities', null=True)


class ProcessLevel(str, Enum):
    ENTERPRISE = "enterprise"
    CORE = "core"
    PROCESS = "process"
    SUBPROCESS = "subprocess"

class Process(TimestampMixin):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    level = fields.CharEnumField(ProcessLevel)
    description = fields.CharField(max_length=255)
    capability = fields.ForeignKeyField('models.Capability', related_name='processes', null=True)