from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
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
IMPORTANT RESPONSE FORMAT:

Always provide a clear, conversational explanation of your findings
Present data insights in complete sentences
When sharing numbers, incorporate them naturally into your response
Avoid bullet points or technical formatting unless specifically requested
Charts should be accompanied by interpretive text explaining key insights

When referring to column names, you MUST use the exact case as shown in the DataFrame. Common columns in this dataset are:

'Model'
'MPG'
'Cylinders'
'Displacement'
'Horsepower'
'Weight'
'Acceleration'
'Year'
'Origin'
'Title'
'Worldwide Gross'
'Production Budget'
'Release Year'
'Content Rating'
'Running Time'
'Genre'
'Creative Type'
'Rotten Tomatoes Rating'
'IMDB Rating'

Before performing any analysis, you MUST:

First examine the DataFrame columns using: print("Available columns:", df.columns.tolist())
Only use columns that actually exist in the DataFrame
If requested columns don't exist, explain conversationally what columns are available instead"""

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

class AnalysisResponse(BaseModel):
    text: str = Field("", description="Analysis results or explanation")
    chart: Optional[Dict[str, Any]] = Field(None, description="Vega-Lite visualization spec")

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
            "description": "Generate a Vega-Lite visualization based on the user's request",
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

def process_tool_calls(
    client: OpenAI,
    initial_response,
    request_data: List[Dict[str, Any]],
    system_prompt: str,
    user_prompt: str
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Process tool calls and generate final response with optional visualization.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    chart_spec = None
    analysis_results = []
    
    # Process all tool calls
    for tool_call in initial_response.choices[0].message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # Execute appropriate tool and collect output
        if func_name == "create_chart":
            df = pd.DataFrame(request_data)
            if args["x_column"] not in df.columns or args["y_column"] not in df.columns:
                tool_result = f"Error: One or more columns not found. Available columns: {df.columns.tolist()}"
            else:
                chart_spec = {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "mark": args["chart_type"],
                    "title": args.get("title", ""),
                    "width": "container",
                    "height": 400,
                    "encoding": {
                        "x": {"field": args["x_column"], "type": "nominal"},
                        "y": {"field": args["y_column"], "type": "quantitative"}
                    }
                }
                if args.get("aggregation"):
                    chart_spec["encoding"]["y"]["aggregate"] = args["aggregation"]
                tool_result = "Chart created successfully"
        
        elif func_name == "analyze_data":
            tool_result = execute_pandas_code(args["code"], request_data)
            analysis_results.append(tool_result)
        
        messages.extend([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            }
        ])
    
    # Get final response incorporating all results
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages + [
            {
                "role": "user",
                "content": "Based on the analysis and visualizations above, provide a clear, "
                          "conversational summary of the insights for the user."
            }
        ]
    )
    
    return final_response.choices[0].message.content, chart_spec, analysis_results

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
            # Initial API call to get tool calls
            initial_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request.prompt}
                ],
                tools=[create_chart_tool(), create_analysis_tool()],
                tool_choice="auto",
                timeout=30
            )
            
            # If no tool calls, return direct response
            if not initial_response.choices[0].message.tool_calls:
                final_response.text = initial_response.choices[0].message.content
                return final_response
            
            # Process tool calls and get final response
            response_text, chart_spec, analysis_results = process_tool_calls(  # Updated to handle 3 return values
                client=client,
                initial_response=initial_response,
                request_data=request.data,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=request.prompt
            )
            
            # If there are analysis results, append them to the response text
            if analysis_results:
                analysis_text = "\n".join(analysis_results)
                response_text = f"{analysis_text}\n\n{response_text}"
            
            final_response.text = response_text
            if chart_spec:
                final_response.chart = chart_spec
            
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)