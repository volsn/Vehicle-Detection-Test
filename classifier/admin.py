from django.contrib import admin
from django.utils.html import format_html
from classifier.models import Camera, Shot

from django.contrib.auth.models import Group
admin.site.unregister(Group)

# Register your models here.
admin.site.index_title = 'Управление Камерами'
admin.site.site_header = 'Панель Администратора'
admin.site.site_title = 'Административный сайт'

@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_filter = ('city', 'active',)
    readonly_fields = ('active', 'stop_camera', 'start_camera',)

    def stop_camera(self, obj):
        return format_html('<a href="/stop/{}">Выключить считывание</a>'\
                .format(obj.pk))
    stop_camera.short_description = 'Выкл'

    def start_camera(self, obj):
        return format_html('<a href="/start/{}">Включить считывание</a>'\
                .format(obj.pk))
    start_camera.short_description = 'Вкл'


@admin.register(Shot)
class ShotAdmin(admin.ModelAdmin):
    list_filter = ('type',)
    readonly_fields = ('display_car_image',)
    list_display = ('timestamp', 'size', 'display_car_image_list',)
    actions = ('change_class_to_civil', 'change_class_to_ambulance',)

    def size(self, obj):
        if not obj.car:
            return 'NaN'
        return '{}x{}'.format(obj.car.width, obj.car.height)

    """
    Methods for changing labels
    """
    def change_class_to_civil(modeladmin, request, queryset):
        queryset.update(type=1, wrong_label=True)
    change_class_to_civil.short_description = 'Изменить класс на "Гражданская"'

    def change_class_to_ambulance(modeladmin, request, queryset):
        queryset.update(type=0, wrong_label=True)
    change_class_to_ambulance.short_description = 'Изменить класс на "Скорая"'

    """
    Methods for displaying images
    """
    def display_full_image(self, obj):
        return format_html('<img src="{url}" width="{width}" height={height} />'.format(
            url = obj.image.url,
            width=obj.image.width // 2,
            height=obj.image.height // 2,
            )
        )

    def display_car_image_list(self, obj):

        if not obj.car:
            return 'NaN'

        # Resizing
        width = obj.car.width
        height = obj.car.height
        if width > height:
            k = width / 200
        else:
            k = height / 200

        width = obj.car.width / k
        height = obj.car.height / k

        return format_html('<img src="{url}" width="{width}" height={height} />'.format(
            url = obj.car.url,
            width=width,
            height=height,
            )
        )
    display_car_image_list.short_description = 'Изображение'

    def display_car_image(self, obj):
        return format_html('<img src="{url}" width="{width}" height={height} />'.format(
            url = obj.car.url,
            width=obj.car.width,
            height=obj.car.height,
            )
        )
