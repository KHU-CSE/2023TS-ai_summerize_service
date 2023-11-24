from django.http import JsonResponse
import requests
import json

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        try:
            question = request.POST.get('question')
            context = request.POST.get('context')

            HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/ainize/klue-bert-base-mrc"
            HUGGING_FACE_API_KEY = "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"
            headers = {
                "Authorization": f"Bearer {HUGGING_FACE_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            payload = {
                "inputs": {
                    "question": question,
                    "context": context
                }
            }

            response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload)
            result = response.json()

            return JsonResponse(result)

        except json.JSONDecodeError as e:
            return JsonResponse({"error": f"Invalid JSON format in the request body: {str(e)}"}, status=400)
        except requests.RequestException as e:
            return JsonResponse({"error": f"Error in making the request to Hugging Face API: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)