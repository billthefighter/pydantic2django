from django.db import models
import uuid

# Models moved from fixtures.py


class LazyRefTarget(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)

    class Meta:
        # app_label = "pydantic2django_testing" # App label set by AppConfig
        db_table = "test_lazy_ref_target"

    def __str__(self):
        return self.name


class LazyRefSource(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)
    # Lazy reference by string
    target = models.ForeignKey(
        "LazyRefTarget",  # String reference
        on_delete=models.CASCADE,
        related_name="sources",
        null=True,
        blank=True,
    )
    # Self-reference (also lazy)
    parent = models.ForeignKey("self", on_delete=models.SET_NULL, related_name="children", null=True, blank=True)

    class Meta:
        # app_label = "pydantic2django_testing" # App label set by AppConfig
        db_table = "test_lazy_ref_source"

    def __str__(self):
        return self.name
