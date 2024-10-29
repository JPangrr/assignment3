from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import pandas as pd
import sys
from io import StringIO
import re
import json
import os
import logging
from typing import Optional, List, Dict, Any, Tuple
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
    allow_origins=[
        "http://localhost:3000",
        "https://jpangrr.github.io",  # Add your GitHub Pages domain
        "https://assignment3-f8gl.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
    Remember to choose the most appropriate chart type based on the data and question. Also, handle any necessary data transformations (such as filtering, aggregation, or binning) that the chart might require.

    Provide only the Vega-Lite JSON spec in your response.
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

def generate_vega_lite_spec(prompt: str, columns_info: Dict[str, str]) -> QueryResponse:
    constructed_prompt = construct_vega_lite_prompt(prompt, columns_info)

    # Step 2: Call OpenAI API to generate Vega-Lite specification
    chat_completion = client.ChatCompletion.create(
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
        model="gpt-4",
    )

    # Try accessing the completion's content correctly
    try:
        vega_lite_spec = chat_completion.choices[0].message['content']
        logger.info(f"Generated Vega-Lite Spec: {vega_lite_spec}")  # Log the Vega-Lite spec
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"KeyError: {str(e)} in response: {chat_completion}")

    # Step 4: Chain another prompt for chart description
    description_prompt = construct_chart_description_prompt(vega_lite_spec)

    description_completion = client.ChatCompletion.create(
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
        model="gpt-4",
    )

    # Try accessing the description content correctly
    try:
        chart_description = description_completion.choices[0].message['content']
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
        
        # Initialize response
        final_response = AnalysisResponse()
        
        try:
            # Modify the initial API call to explicitly allow multiple tool calls
            initial_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\nYou can and should use multiple tools when appropriate. If a user asks for both visualization and analysis, use both the create_chart and analyze_data tools."},
                    {"role": "user", "content": request.prompt}
                ],
                tools=[create_chart_tool(), create_analysis_tool()],
                tool_choice="auto",
                timeout=30
            )
            
            # Log initial response for debugging
            logger.info(f"Initial response: {initial_response}")
            
            # If no tool calls, return direct response
            if not initial_response.choices[0].message.tool_calls:
                final_response.text = initial_response.choices[0].message.content
                return final_response
            
            # Process tool calls and get final response
            response_text, chart_spec, summary, analysis_results = process_tool_calls(
                client=client,
                initial_response=initial_response,
                request_data=request.data,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=request.prompt
            )
            
            # Combine all results into the final response
            final_text_parts = []
            
            # Add analysis results if present
            if analysis_results:
                final_text_parts.extend(analysis_results)
            
            # Add response text
            if response_text:
                final_text_parts.append(response_text)
            
            final_response.text = "\n\n".join(final_text_parts)
            if chart_spec:
                final_response.chart = chart_spec
            if summary:
                final_response.summary = summary
            
            return final_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request with OpenAI API: {str(e)}"
            )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

def process_tool_calls(
    client: OpenAI,
    initial_response,
    request_data: List[Dict[str, Any]],
    system_prompt: str,
    user_prompt: str
) -> Tuple[str, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], List[str]]:
    """
    Process tool calls and generate final response with optional visualization.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    chart_spec = None
    summary = None
    analysis_results = []
    tool_results = []
    
    # Process all tool calls
    for tool_call in initial_response.choices[0].message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        logger.info(f"Processing tool call: {func_name} with arguments: {args}")
        
        tool_result = None
        
        # Execute appropriate tool and collect output
        if func_name == "create_chart":
            df = pd.DataFrame(request_data)
            if args["x_column"] not in df.columns or args["y_column"] not in df.columns:
                tool_result = f"Error: One or more columns not found. Available columns: {df.columns.tolist()}"
            else:
                columns_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
                query_response = generate_vega_lite_spec(user_prompt, columns_info)
                chart_spec = json.loads(query_response.vega_lite_spec)
                tool_result = {
                    "chart": chart_spec,
                    "description": query_response.chart_description
                }
        
        elif func_name == "analyze_data":
            analysis_result = execute_pandas_code(args["code"], request_data)
            analysis_results.append(analysis_result)
            tool_result = analysis_result
        
        # Store the tool result and add to messages
        tool_results.append({
            "tool_call_id": tool_call.id,
            "result": tool_result
        })
        
        messages.extend([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            }
        ])
    
    # Get final response incorporating all results
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages + [
            {
                "role": "user",
                "content": "Based on all the analysis and visualizations above, provide a clear, "
                          "conversational summary of the insights for the user."
            }
        ]
    )
    
    return final_response.choices[0].message.content, chart_spec, summary, analysis_results

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))  # Use PORT from environment or default to 10000
    uvicorn.run(app, host="0.0.0.0", port=port)