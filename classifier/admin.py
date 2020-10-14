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
    readonly_fields = ('online_view',)

    def online_view(self, obj):
        return format_html('<a href="/visualize/{}">Онлайн просмотр</a>'.format(obj.pk))
        #.format(str(obj.id))
    online_view.short_description = 'Камера'


@admin.register(Shot)
class ShotAdmin(admin.ModelAdmin):
    pass
