from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import pandas as pd
import sys
from io import StringIO
import json
import os
import logging
from typing import Optional, List, Dict, Any, Tuple, NamedTuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

SYSTEM_PROMPT = """You are a data analysis assistant specialized in both visualization and statistical analysis. Your role is to help users understand their data through clear explanations, appropriate visualizations, and insightful analysis.

IMPORTANT: You can and should use multiple tools when appropriate. If a user asks for both visualization and analysis, use both the create_chart and analyze_data tools.

IMPORTANT RESPONSE FORMAT:

Always provide a clear, conversational explanation of your findings
Present data insights in complete sentences
When sharing numbers, incorporate them naturally into your response
Avoid bullet points or technical formatting unless specifically requested
Charts should be accompanied by interpretive text explaining key insights

WHEN TO USE THE CREATE ANALYSIS TOOL:

Use the create_analysis tool when the user requests a specific data analysis task that requires executing pandas code (e.g. calculating summary statistics, filtering/transforming the data, etc.)
Example: "Calculate the average MPG for cars with more than 100 horsepower"
Make sure to use print() statements in your pandas code to return the results in a formatted way for the user

WHEN TO USE THE CREATE CHART TOOL:

Use the create_chart tool when the user requests a specific visualization of the data
Example: "Create a bar chart showing the distribution of car models by origin"
Provide a title, chart type, x-axis, and y-axis, and optionally an aggregation method
Accompany the chart with a conversational explanation of the key insights it reveals

COLUMN REFERENCE:
The common columns in this dataset are:
'Model', 'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Year', 'Origin', 'Title', 'Worldwide Gross', 'Production Budget', 'Release Year', 'Content Rating', 'Running Time', 'Genre', 'Creative Type', 'Rotten Tomatoes Rating', 'IMDB Rating'
Before performing any analysis, you MUST:

Examine the DataFrame columns using: print("Available columns:", df.columns.tolist())
Only use columns that actually exist in the DataFrame
If requested columns don't exist, explain conversationally what columns are available instead

WHEN TO USE BOTH TOOLS:

If the user requests both a chart and a summary table, use both the create_chart and create_analysis tools.
Example: "Show a breakdown of cars by their origin, as a bar chart and a summary table."

"""

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    raise

class QueryRequest(BaseModel):
    prompt: str = Field(..., description="Analysis prompt or question")
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze")

class QueryResponse(BaseModel):
    vega_lite_spec: str
    chart_description: str

class AnalysisResponse(BaseModel):
    text: str = Field("", description="Analysis results or explanation")
    chart: Optional[Dict[str, Any]] = Field(None, description="Vega-Lite visualization spec")
    summary: Optional[List[Dict[str, Any]]] = Field(None, description="Summary table")

class ColumnInfo(NamedTuple):
    name: str
    type: str
    sample: Any

def validate_data(data: List[Dict[str, Any]]) -> None:
    """Validate the input data structure and log column information."""
    if not data:
        raise ValueError("Empty data provided")
    if not isinstance(data[0], dict):
        raise ValueError("Invalid data format - expected list of dictionaries")
    
    df = pd.DataFrame(data)
    logger.info(f"DataFrame created with shape: {df.shape}")
    logger.info(f"Available columns: {df.columns.tolist()}")
    logger.info(f"Column types: {df.dtypes.to_dict()}")

def construct_vega_lite_prompt(user_question, columns_info):
    # Prepare dataset information for the prompt with column names, types, and sample values
    columns = [
        f"{col.name} (Type: {col.type}, Sample: {col.sample})" for col in columns_info
    ]
    
    # Construct the prompt for Vega-Lite JSON specification generation
    prompt = f"""
    You are a helpful data science assistant that generates accurate and valid Vega-Lite JSON specifications from user questions and dataset information. You should have a valid JSON specification each time.

    Based on the following dataset information:
    
    Columns: {', '.join(columns)}

    Please generate a valid Vega-Lite JSON specification for the following question: "{user_question}"
    
    Remember to:
    1. Choose the most appropriate chart type based on the data and question
    2. Do NOT include the "data" field or "values" - these will be added separately
    3. Handle any necessary data transformations (filtering, aggregation, binning)
    4. Ensure all referenced columns exist in the dataset
    5. Use proper encoding channels (x, y, color, etc.)
    6. Set appropriate scales and axes based on data types

    Provide only the Vega-Lite JSON spec in your response, with no additional text or explanation.
    """
    return prompt

def construct_chart_description_prompt(vega_lite_spec):
    # Construct the prompt to generate a description of the Vega-Lite chart
    prompt = f"""
    You are a helpful assistant that explains data visualizations clearly.

    Based on the following Vega-Lite chart specification, provide a simple and clear description (one to two sentences) of the chart and what insights it conveys:

    Vega-Lite Spec: {vega_lite_spec}

    """
    
    return prompt

def generate_vega_lite_spec(prompt: str, columns_info: List[ColumnInfo]) -> QueryResponse:
    constructed_prompt = construct_vega_lite_prompt(prompt, columns_info)

    # Step 2: Call OpenAI API to generate Vega-Lite specification
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful data science assistant that generates accurate Vega-Lite specifications from user questions and dataset information.",
            },
            {
                "role": "user",
                "content": constructed_prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    # Try accessing the completion's content correctly
    try:
        vega_lite_spec = chat_completion.choices[0].message.content.strip()
        logger.info(f"Generated Vega-Lite Spec: {vega_lite_spec}")  # Log the Vega-Lite spec
        
        # Validate JSON
        try:
            json.loads(vega_lite_spec)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from markdown code block
            if "```json" in vega_lite_spec:
                vega_lite_spec = vega_lite_spec.split("```json")[1].split("```")[0].strip()
            elif "```" in vega_lite_spec:
                vega_lite_spec = vega_lite_spec.split("```")[1].strip()
                
            # Validate extracted JSON
            json.loads(vega_lite_spec)
            
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"KeyError: {str(e)} in response: {chat_completion}")

    # Step 4: Chain another prompt for chart description
    description_prompt = construct_chart_description_prompt(vega_lite_spec)

    description_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that explains data visualizations clearly.",
            },
            {
                "role": "user",
                "content": description_prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    # Try accessing the description content correctly
    try:
        chart_description = description_completion.choices[0].message.content
        logger.info(f"Generated Chart Description: {chart_description}")  # Log the chart description
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"KeyError: {str(e)} in response: {description_completion}")

    # Step 6: Return both Vega-Lite specification and description
    return QueryResponse(
        vega_lite_spec=vega_lite_spec,
        chart_description=chart_description,
    )

def execute_pandas_code(code: str, data: List[Dict[str, Any]]) -> str:
    """Execute pandas code with enhanced error handling and column validation."""
    logger.info(f"Executing pandas code: {code}")

    old_stdout = sys.stdout
    mystdout = StringIO()
    sys.stdout = mystdout
    
    try:
        df = pd.DataFrame(data)
        logger.info(f"DataFrame created with shape: {df.shape}")
        
        columns_list = df.columns.tolist()
        
        globals_dict = {
            "df": df,
            "pd": pd,
            "__builtins__": {
                name: __builtins__[name]
                for name in ['print', 'len', 'range', 'sum', 'min', 'max', 'round']
            }
        }
        
        cleaned_code = code.strip().strip('`').strip()
        exec(cleaned_code, globals_dict)
        
        return mystdout.getvalue()
    except KeyError as e:
        logger.error(f"Column not found error: {e}")
        return f"Error: Column {str(e)} not found. Available columns are: {columns_list}"
    except Exception as e:
        logger.error(f"Error executing pandas code: {e}")
        return f"Error executing analysis: {str(e)}"
    finally:
        sys.stdout = old_stdout

def create_chart_tool() -> Dict[str, Any]:
    """Create the Vega-Lite chart creation tool specification."""
    return {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Generate a Vega-Lite visualization based on the user's request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "area", "point"],
                        "description": "The type of chart to create"
                    },
                    "x_column": {
                        "type": "string",
                        "description": "The column to use for x-axis"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "The column to use for y-axis"
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "median", "min", "max"],
                        "description": "Aggregation method if needed"
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title"
                    }
                },
                "required": ["chart_type", "x_column", "y_column"]
            }
        }
    }

def create_analysis_tool() -> Dict[str, Any]:
    """Create the pandas data analysis tool specification."""
    return {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Execute pandas code to analyze the data. Always use print() to show results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code using pandas (df is the DataFrame name)"
                    }
                },
                "required": ["code"]
            }
        }
    }

@app.post("/query", response_model=AnalysisResponse)
async def process_query(request: QueryRequest) -> AnalysisResponse:
    """Process a data analysis query with enhanced tool calling and response handling."""
    logger.info("Processing new query request")
    
    try:
        # Validate input data and log column information
        validate_data(request.data)
        
        # Initialize response and message history
        final_response = AnalysisResponse()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt}
        ]
        
        # Initialize tool results storage
        chart_spec = None
        summary = None
        analysis_results = []
        
        # React loop - continue until assistant provides final response without tool calls
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # Get next action from assistant
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    tools=[create_chart_tool(), create_analysis_tool()],
                    tool_choice="auto"
                )
                
                assistant_message = response.choices[0].message
                
                # If no tool calls, we have our final response
                if not assistant_message.tool_calls:
                    final_response.text = assistant_message.content
                    if chart_spec:
                        final_response.chart = chart_spec
                    if summary:
                        final_response.summary = summary
                    break
                
                # Process tool calls
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    # Execute appropriate tool
                    if func_name == "create_chart":
                        df = pd.DataFrame(request.data)
                        columns_info = [
                            ColumnInfo(
                                name=col,
                                type=str(dtype),
                                sample=str(df[col].iloc[0]) if len(df) > 0 else None
                            )
                            for col, dtype in df.dtypes.items()
                        ]
                        query_response = generate_vega_lite_spec(request.prompt, columns_info)
                        chart_spec = json.loads(query_response.vega_lite_spec)
                        # Add the data to the chart specification
                        chart_spec["data"] = {"values": request.data}
                        tool_result = query_response.chart_description
                    
                    elif func_name == "analyze_data":
                        tool_result = execute_pandas_code(args["code"], request.data)
                        analysis_results.append(tool_result)
                    
                    # Add tool result to message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in react loop iteration {iteration}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error in analysis iteration {iteration}: {str(e)}"
                )
        
        if iteration >= max_iterations:
            logger.warning("Reached maximum iterations in react loop")
            final_response.text = "Analysis completed but reached maximum iterations. Results may be incomplete."
        
        return final_response
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)