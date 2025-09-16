from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("tests", "0008_childtype_parenttype_itemtype"),
    ]

    operations = [
        migrations.AlterField(
            model_name="itemtype",
            name="parenttype",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="items",
                to="tests.parenttype",
                null=False,
            ),
        ),
    ]
