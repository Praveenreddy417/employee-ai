from rest_framework.decorators import api_view
# from rest_framework.response import Response
from .rag import ask_question

# @api_view(["POST"])
# def chat_view(request):
#     message = request.data.get("message")
#     answer = ask_question(message)
#     return Response({"response": answer})

from drf_spectacular.utils import extend_schema
from .serializers import ChatRequestSerializer, ChatResponseSerializer
from rest_framework.response import Response

@extend_schema(
    request=ChatRequestSerializer,
    responses=ChatResponseSerializer
)
@api_view(["POST"])
def chat_view(request):
    serializer = ChatRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    message = serializer.validated_data["message"]
    answer = ask_question(message)

    return Response({"response": answer})


# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from .rag import ask_question
# from .serializers import ChatRequestSerializer

# @api_view(["POST"])
# def chat_view(request):
#     serializer = ChatRequestSerializer(data=request.data)
#     serializer.is_valid(raise_exception=True)

#     message = serializer.validated_data["message"]
#     answer = ask_question(message)

#     return Response({"response": answer})
