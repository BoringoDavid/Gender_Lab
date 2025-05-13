from django.contrib import admin
from django.contrib import admin, messages
from django.utils.html import format_html
from django.urls import path, reverse
from django.http import HttpResponseRedirect
import requests
from django.contrib import admin
from .models import *

# Admin panel titles
admin.site.site_header = "Warehouse Management Admin"
admin.site.site_title = "Warehouse Admin Portal"
admin.site.index_title = "Welcome to the Warehouse Admin Dashboard"

# Custom action to sync selected files to server
@admin.action(description='Send to server')
def sync_selected_to_server(modeladmin, request, queryset):
    api_url = "https://yourserver.com/api/sync/"  # Replace with your real API endpoint

    for obj in queryset:
        data = {
            "id": obj.id,
            "file_type": obj.file_type,
            "source_type": obj.source_type,
            "uploaded_at": obj.uploaded_at.isoformat(),
            "start_date": obj.start_date.isoformat() if obj.start_date else None,
            "end_date": obj.end_date.isoformat() if obj.end_date else None,
        }

        try:
            response = requests.post(api_url, json=data)
            if response.status_code == 200:
                messages.success(request, f"File {obj.id} sent to server successfully.")
            else:
                messages.error(request, f"Failed to send file {obj.id}. Status: {response.status_code}")
        except requests.RequestException as e:
            messages.error(request, f"Error sending file {obj.id}: {e}")

@admin.register(DataFileUpload)
class DataFileUploadAdmin(admin.ModelAdmin):
    list_display = (
        'file_link',
        'file_type',
        'source_type',
        'uploaded_at',
        'start_date',
        'end_date',
    )
    list_filter = ('file_type', 'source_type', 'uploaded_at')
    search_fields = ('file', 'survey_name', 'external_file_name')
    ordering = ('-uploaded_at',)
    actions = [sync_selected_to_server]

    def file_link(self, obj):
        if obj.file:
            return format_html('<a href="{}" target="_blank">{}</a>', obj.file.url, obj.file.name)
        return "-"
    file_link.short_description = 'File'

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context['title'] = "Uploaded Data Files Overview"
        extra_context['sync_all_url'] = reverse('admin:sync_all_files')
        return super().changelist_view(request, extra_context=extra_context)

    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = extra_context or {}
        extra_context['title'] = "Edit Uploaded File Details"
        return super().change_view(request, object_id, form_url, extra_context=extra_context)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path("sync-all/", self.admin_site.admin_view(self.sync_all_files), name="sync_all_files"),
        ]
        return custom_urls + urls

    def sync_all_files(self, request):
        api_url = "https://yourserver.com/api/sync/"  # Replace with your real API endpoint

        for obj in DataFileUpload.objects.all():
            data = {
                "id": obj.id,
                "file_type": obj.file_type,
                "source_type": obj.source_type,
                "uploaded_at": obj.uploaded_at.isoformat(),
                "start_date": obj.start_date.isoformat() if obj.start_date else None,
                "end_date": obj.end_date.isoformat() if obj.end_date else None,
            }

            try:
                response = requests.post(api_url, json=data)
                if response.status_code != 200:
                    messages.error(request, f"Failed to sync {obj.id}: {response.status_code}")
            except Exception as e:
                messages.error(request, f"Error syncing {obj.id}: {e}")

        messages.success(request, "All records synced.")
        return HttpResponseRedirect("../")

@admin.register(Survey)
class SurveyAdmin(admin.ModelAdmin):
    list_display = ('survey_id', 'survey_name', 'year','file')
    search_fields = ('survey_name',)
    list_filter = ('year',)

@admin.register(Indicator)
class IndicatorAdmin(admin.ModelAdmin):
    list_display = ('indicator_name', 'survey')
    search_fields = ('indicator_name',)
    list_filter = ('survey',)




