from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Document
import textract

@csrf_exempt
def upload_document(request):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')
        combined_text = ""
        document_ids = []

        for uploaded_file in uploaded_files:
            document = Document(file=uploaded_file)
            document.save()
            text = textract.process(document.file.path).decode('utf-8')
            combined_text += text + " "
            document.converted_text = text
            document.converted = True
            document.save()

            document_ids.append(document.id)

        return JsonResponse({'message': 'Files uploaded and combined successfully', 'combined_text': combined_text})

