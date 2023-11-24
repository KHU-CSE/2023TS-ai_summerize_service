from django.contrib import admin
from .models import Document

class DocumentAdmin(admin.ModelAdmin):
    list_display = ['file', 'converted_text','converted']
    search_fields = ['file']


admin.site.register(Document, DocumentAdmin)