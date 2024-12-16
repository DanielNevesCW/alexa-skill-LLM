import logging, os, json, traceback, boto3
from openai import OpenAI
import ask_sdk_core.utils as ask_utils
from ask_sdk_core.skill_builder import CustomSkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_dynamodb.adapter import DynamoDbAdapter

# Initialize services
def init_services():
    """Initialize DynamoDB and OpenAI services"""
    ddb = DynamoDbAdapter(
        table_name=os.environ.get('DYNAMODB_PERSISTENCE_TABLE_NAME'),
        create_table=False,
        dynamodb_resource=boto3.resource('dynamodb', region_name=os.environ.get('DYNAMODB_PERSISTENCE_REGION'))
    )
    openai_client = OpenAI(api_key="OPENAI_API_KEY") # TODO: Pass this from env
    return ddb, openai_client

dynamodb_adapter, client = init_services()

class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        persistent_attrs = handler_input.attributes_manager.persistent_attributes
        
        if not persistent_attrs:
            assistant = client.beta.assistants.create(
                instructions=""""You are an AI English language tutor designed to teach English to non-native speakers. Your goal is to provide clear, engaging, and effective English lessons tailored to the student's proficiency level. !!IMPORTANT Answer with 2 lines max with 30 letters in max""",
                model="gpt-4o",
            )
            persistent_attrs["assistant_id"] = assistant.id
        
        thread = client.beta.threads.create()
        persistent_attrs["thread_id"] = thread.id
        handler_input.attributes_manager.save_persistent_attributes()
        
        return handler_input.response_builder.speak("Chat GPT mode activated.").ask("Chat GPT mode activated.").response

class GptQueryIntentHandler(AbstractRequestHandler):
    """Handler for GPT queries"""
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        try:
            query = handler_input.request_envelope.request.intent.slots["query"].value
            response = self.generate_gpt_response(query, handler_input)
            return handler_input.response_builder.speak(response).ask("...").response
        except Exception:
            return handler_input.response_builder.speak("Sorry, try again.").ask("Please try again.").response

    def generate_gpt_response(self, query: str, handler_input: HandlerInput) -> str:
        """Generate GPT response with error handling"""
        try:
            attrs = handler_input.attributes_manager.persistent_attributes
            thread_id, assistant_id = attrs.get("thread_id"), attrs.get("assistant_id")
            
            if not all([thread_id, assistant_id]):
                return "Sorry, I lost my memory. Please restart."
            
            client.beta.threads.messages.create(thread_id=thread_id, role="user", content=query)
            run = client.beta.threads.runs.create_and_poll(thread_id=thread_id, assistant_id=assistant_id)
            
            if run.status == 'completed':
                return self.get_assistant_response(thread_id)
            
            return "Something unexpected happened. Let's restart."
        except Exception as e:
            logging.error(f"Critical error: {str(e)}")
            return "Technical difficulties. Try again later."

    def get_assistant_response(self, thread_id: str) -> str:
        """Get latest assistant response"""
        try:
            return client.beta.threads.messages.list(thread_id=thread_id).data[0].content[0].text.value.strip()
        except Exception as e:
            logging.error(f"Response error: {str(e)}")
            return "Can't speak right now. Try again."

class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors."""
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logging.error(f"Error: {exception}")
        return (
            handler_input.response_builder
                .speak("Sorry, I had trouble processing your request. Please try again.")
                .ask("Please try again.")
                .response
        )

# Setup and Lambda handler
sb = CustomSkillBuilder(persistence_adapter=dynamodb_adapter)
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

def lambda_handler(event, context):
    return sb.lambda_handler()(event, context)