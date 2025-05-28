import openai
from openai import OpenAI

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Iterator
from enum import Enum
import sys


class OutputFormat(Enum):
    """Output format options for pretty printing"""
    TABLE = "table"
    JSON = "json"
    DETAILED = "detailed"


class BatchPrettyPrinter:
    """
    Utility class to pretty print the results from OpenAI's client.batches.list()
    
    This class provides multiple output formats and filtering options for better
    visualization of batch job data.
    """
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the pretty printer
        
        Args:
            use_colors: Whether to use ANSI colors in output (default: True)
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        
        # ANSI color codes
        if self.use_colors:
            self.colors = {
                'green': '\033[92m',
                'yellow': '\033[93m',
                'blue': '\033[94m',
                'red': '\033[91m',
                'bold': '\033[1m',
                'end': '\033[0m'
            }
        else:
            self.colors = {k: '' for k in ['green', 'yellow', 'blue', 'red', 'bold', 'end']}
    
    def _color_text(self, text: str, color_key: str) -> str:
        """Apply color to text if colors are enabled"""
        if not self.use_colors or color_key not in self.colors:
            return text
        return f"{self.colors[color_key]}{text}{self.colors['end']}"
    
    def _format_timestamp(self, timestamp: Optional[Union[int, float]]) -> str:
        """Convert a timestamp to a human-readable format"""
        if not timestamp:
            return "N/A"
        
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError):
            return str(timestamp)
    
    def _get_status_color(self, status: str) -> str:
        """Get the appropriate color for a status"""
        status_colors = {
            'completed': 'green',
            'in_progress': 'yellow',
            'validating': 'blue',
            'failed': 'red',
        }
        return status_colors.get(status.lower(), '')
    
    def _format_batch_as_dict(self, batch_obj) -> Dict[str, Any]:
        """
        Convert a Batch object to a dictionary with formatted fields
        
        Args:
            batch_obj: A Batch object from the OpenAI API
            
        Returns:
            Dict: A dictionary with formatted fields
        """
        # Create a base dictionary with all attributes
        batch_dict = {}
        
        # Try to convert to dict if the object supports it
        try:
            if hasattr(batch_obj, 'model_dump'):  # Pydantic v2
                batch_dict = batch_obj.model_dump()
            elif hasattr(batch_obj, 'dict'):  # Pydantic v1
                batch_dict = batch_obj.dict()
            else:
                # Fallback to __dict__ with filtering
                batch_dict = {k: v for k, v in batch_obj.__dict__.items() 
                             if not k.startswith('_') and not callable(v)}
        except Exception:
            # If all else fails, just use dir() to get attributes
            for attr in dir(batch_obj):
                if not attr.startswith('_') and not callable(getattr(batch_obj, attr, None)):
                    batch_dict[attr] = getattr(batch_obj, attr, None)
        
        # Format timestamps for display
        for ts_field in ['created_at', 'completed_at', 'expires_at']:
            if ts_field in batch_dict and batch_dict[ts_field]:
                batch_dict[f"{ts_field}_formatted"] = self._format_timestamp(batch_dict[ts_field])
        
        return batch_dict
    
    def _extract_batch_fields(self, batch) -> Dict[str, Any]:
        """Extract relevant fields from a batch object"""
        batch_dict = self._format_batch_as_dict(batch)
        
        # Extract common fields with defaults
        return {
            'id': batch_dict.get('id', 'N/A'),
            'status': batch_dict.get('status', 'unknown'),
            'created_at': batch_dict.get('created_at'),
            'created_at_formatted': batch_dict.get('created_at_formatted', 'N/A'),
            'completed_at': batch_dict.get('completed_at'),
            'completed_at_formatted': batch_dict.get('completed_at_formatted', 'N/A'),
            'endpoint': batch_dict.get('endpoint', 'N/A'),
            'input_file_id': batch_dict.get('input_file_id', 'N/A'),
            'output_file_id': batch_dict.get('output_file_id', 'N/A'),
            'error_file_id': batch_dict.get('error_file_id', 'N/A'),
            'expires_at': batch_dict.get('expires_at'),
            'expires_at_formatted': batch_dict.get('expires_at_formatted', 'N/A'),
            'metadata': batch_dict.get('metadata', {}),
            'completion_window': batch_dict.get('completion_window', 'N/A'),
            'full_obj': batch_dict
        }
    
    def print_table(self, batches: Iterator, max_width: int = 120) -> None:
        """
        Print batches in a table format
        
        Args:
            batches: Iterator of Batch objects from client.batches.list()
            max_width: Maximum width of the table
        """
        count = 0
        collected_batches = []
        
        # Collect batch data first to determine column widths
        for batch in batches:
            count += 1
            batch_data = self._extract_batch_fields(batch)
            collected_batches.append(batch_data)
        
        if count == 0:
            print("No batch jobs found.")
            return
        
        # Define columns to display
        columns = [
            ('ID', 'id', 36),
            ('Status', 'status', 12),
            ('Created', 'created_at_formatted', 22),
            ('Endpoint', 'endpoint', 20),
        ]
        
        # Calculate actual column widths (not exceeding max_width)
        total_static_width = 3 * (len(columns) - 1)  # Space for separators
        available_width = max_width - total_static_width
        
        # Calculate column widths
        for idx, (header, key, default_width) in enumerate(columns):
            max_content_width = max(
                len(header),
                max((len(str(batch[key])) for batch in collected_batches), default=0)
            )
            columns[idx] = (header, key, min(max_content_width, default_width))
        
        # Adjust column widths to fit max_width
        total_width = sum(width for _, _, width in columns) + total_static_width
        if total_width > max_width:
            # Proportionally reduce widths
            excess = total_width - max_width
            total_default = sum(width for _, _, width in columns)
            for idx, (header, key, width) in enumerate(columns):
                reduction = int((width / total_default) * excess)
                adjusted_width = max(5, width - reduction)  # Ensure minimum width
                columns[idx] = (header, key, adjusted_width)
        
        # Print header
        header_row = ' | '.join(
            self._color_text(header.ljust(width), 'bold')
            for header, _, width in columns
        )
        print(header_row)
        print('-' * min(len(header_row), max_width))
        
        # Print data rows
        for batch in collected_batches:
            status_color = self._get_status_color(batch['status'])
            
            row_parts = []
            for header, key, width in columns:
                value = str(batch[key])
                if len(value) > width:
                    value = value[:width-3] + '...'
                
                # Color the status column
                if key == 'status' and status_color:
                    value = self._color_text(value.ljust(width), status_color)
                else:
                    value = value.ljust(width)
                
                row_parts.append(value)
            
            print(' | '.join(row_parts))
        
        print(f"\nTotal batch jobs: {count}")

    def print_detailed(self, batches: Iterator) -> None:
        """
        Print batches in a detailed format
        
        Args:
            batches: Iterator of Batch objects from client.batches.list()
        """
        count = 0
        print("=" * 80)
        print(self._color_text("BATCH JOBS".center(80), 'bold'))
        print("=" * 80)
        
        for batch in batches:
            count += 1
            batch_data = self._extract_batch_fields(batch)
            
            # Format status with color
            status = batch_data['status']
            status_color = self._get_status_color(status)
            
            # Prepare metadata string if present
            metadata_str = "None"
            if batch_data['metadata'] and isinstance(batch_data['metadata'], dict):
                metadata_items = [f"{k}: {v}" for k, v in batch_data['metadata'].items()]
                metadata_str = ", ".join(metadata_items)
            
            print(f"{self._color_text('Batch #' + str(count), 'bold')}:")
            print(f"  ID:                {batch_data['id']}")
            print(f"  Status:            {self._color_text(status, status_color)}")
            print(f"  Created:           {batch_data['created_at_formatted']}")
            print(f"  Completed:         {batch_data['completed_at_formatted']}")
            print(f"  Expires:           {batch_data['expires_at_formatted']}")
            print(f"  Endpoint:          {batch_data['endpoint']}")
            print(f"  Completion Window: {batch_data['completion_window']}")
            print(f"  Input File ID:     {batch_data['input_file_id']}")
            print(f"  Output File ID:    {batch_data['output_file_id']}")
            print(f"  Error File ID:     {batch_data['error_file_id']}")
            print(f"  Metadata:          {metadata_str}")
            print("-" * 80)
        
        if count == 0:
            print("No batch jobs found.")
        
        print(f"Total batch jobs: {count}")
        print("=" * 80)

    def to_json(self, batches: Iterator, indent: int = 2) -> str:
        """
        Convert batches to a JSON string with formatting
        
        Args:
            batches: Iterator of Batch objects from client.batches.list()
            indent: JSON indentation level
            
        Returns:
            str: Formatted JSON string
        """
        batch_list = []
        
        for batch in batches:
            batch_data = self._extract_batch_fields(batch)
            batch_list.append(batch_data['full_obj'])
        
        return json.dumps(batch_list, indent=indent, default=str)

    def print_json(self, batches: Iterator, indent: int = 2) -> None:
        """
        Print batches as formatted JSON
        
        Args:
            batches: Iterator of Batch objects from client.batches.list()
            indent: JSON indentation level
        """
        print(self.to_json(batches, indent))

    def print(self, batches: Iterator, format: OutputFormat = OutputFormat.TABLE) -> None:
        """
        Print batches in the specified format
        
        Args:
            batches: Iterator of Batch objects from client.batches.list()
            format: Output format (table, json, or detailed)
        """
        if format == OutputFormat.TABLE:
            self.print_table(batches)
        elif format == OutputFormat.JSON:
            self.print_json(batches)
        elif format == OutputFormat.DETAILED:
            self.print_detailed(batches)
        else:
            raise ValueError(f"Unknown format: {format}")


# Example usage
def pretty_print_batches(batches, format="table", use_colors=True):
    """
    Pretty print OpenAI batch jobs
    
    Args:
        batches: The result from client.batches.list()
        format: Output format ("table", "json", or "detailed")
        use_colors: Whether to use ANSI colors in output
    """
    printer = BatchPrettyPrinter(use_colors=use_colors)
    
    if format.lower() == "json":
        printer.print(batches, OutputFormat.JSON)
    elif format.lower() == "detailed":
        printer.print(batches, OutputFormat.DETAILED)
    else:
        printer.print(batches, OutputFormat.TABLE)



client = OpenAI()

print(f"openai key: {client.api_key}")
print(f"openai organization: {client.organization}")

## list all batches
pretty_print_batches(client.batches.list(), format="detailed", use_colors=True)

for batch in client.batches.list():
    if batch.metadata is not None:
        if "model" in batch.metadata and batch.status == "completed":
            print(f"Batch ID: {batch.id}, Model: {batch.metadata['model']}")