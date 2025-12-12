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
    description = fields.TextField()
    domain = fields.ForeignKeyField('models.Domain', related_name='capabilities', null=True)


class ProcessLevel(str, Enum):
    ENTERPRISE = "enterprise"
    CORE = "core"
    PROCESS = "process"

class Process(TimestampMixin):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    level = fields.CharEnumField(ProcessLevel)
    description = fields.TextField()
    capability = fields.ForeignKeyField('models.Capability', related_name='processes', null=True)
    category = fields.CharField(max_length=255, null=True)

class SubProcess(TimestampMixin):
  id = fields.IntField(pk=True)
  name = fields.CharField(max_length=255)
  description = fields.TextField(null=True)
  process = fields.ForeignKeyField('models.Process', related_name='subprocesses', null=False)
  category = fields.CharField(max_length=255, null=True)

class LLMSettings(TimestampMixin):
  id = fields.IntField(pk=True)
  provider = fields.CharField(max_length=50, default="secure")
  vault_name = fields.CharField(max_length=255, default="https://kvcapabilitycompass.vault.azure.net/")
  temperature = fields.FloatField(default=0.2)
  max_tokens = fields.IntField(default=1500)
  top_p = fields.FloatField(default=0.9)
  
  class Meta:
    table = "llm_settings"