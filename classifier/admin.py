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
    readonly_fields = ('online_view', 'active', 'stop_camera', 'start_camera',)

    def online_view(self, obj):
        return format_html('<a href="/visualize/{}">Онлайн просмотр</a>'\
                .format(obj.pk))
    online_view.short_description = 'Камера'

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
    readonly_fields = ('proba',)
