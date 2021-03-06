# Generated by Django 2.2.7 on 2020-10-12 09:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0008_auto_20201012_0856'),
    ]

    operations = [
        migrations.AlterField(
            model_name='camera',
            name='ip_adress',
            field=models.URLField(max_length=124, verbose_name='IP Адрес'),
        ),
        migrations.AlterField(
            model_name='image',
            name='timestamp',
            field=models.CharField(editable=False, max_length=256, verbose_name='Информация'),
        ),
    ]
