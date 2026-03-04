#!/usr/bin/env python3
"""
MLIR Stream Parser - Extract dispatch dependencies and build DAG
Parses IREE MLIR stream files to create a dependency graph of dispatch operations.
Handles concurrent execution blocks to identify parallel dispatches.
"""

import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class DispatchNode:
    """Represents a dispatch operation in the MLIR stream"""
    name: str
    dispatch_id: int
    workgroup_export: str = ""
    operations: List[str] = field(default_factory=list)
    input_tensors: List[str] = field(default_factory=list)
    output_tensors: List[str] = field(default_factory=list)
    dependencies: Set[int] = field(default_factory=set)
    concurrent_group: Optional[int] = None  # Stage number where this dispatch can run
    execution_order: int = 0  # Sequential order of execution stages
    metadata: Dict = field(default_factory=dict)


class MLIRStreamParser:
    """Parse MLIR stream files and extract dispatch dependencies with concurrent execution info"""
    
    def __init__(self, mlir_file_path: str):
        self.mlir_file_path = Path(mlir_file_path)
        self.content = ""
        self.dispatches: Dict[int, DispatchNode] = {}
        self.graph = nx.DiGraph()
        self.concurrent_groups: List[Set[int]] = []  # Groups of dispatches that can run concurrently
        self.execution_stages: List[List[int]] = []  # Sequential stages of execution
        
    def load_file(self):
        """Load MLIR file content"""
        with open(self.mlir_file_path, 'r') as f:
            self.content = f.read()
    
    def parse_dispatches(self):
        """Extract all dispatch operations from the MLIR file"""
        # Pattern to match stream.executable blocks
        executable_pattern = r'stream\.executable private @(\w+\$async_dispatch_(\d+))\s*\{(.*?)\n\s*\}'
        
        matches = re.finditer(executable_pattern, self.content, re.DOTALL)
        
        for match in matches:
            full_name = match.group(1)
            dispatch_id = int(match.group(2))
            body = match.group(3)
            
            node = DispatchNode(
                name=full_name,
                dispatch_id=dispatch_id
            )
            
            # Extract workgroup export
            export_match = re.search(r'stream\.executable\.export public @(\S+)', body)
            if export_match:
                node.workgroup_export = export_match.group(1)
            
            # Extract operations from builtin.module
            ops_match = re.search(r'builtin\.module\s*\{(.*?)\}', body, re.DOTALL)
            if ops_match:
                ops_body = ops_match.group(1)
                node.operations = self._extract_operations(ops_body)
            
            # Extract tensor information
            node.input_tensors, node.output_tensors = self._extract_tensors(body)
            
            self.dispatches[dispatch_id] = node
    
    def _extract_operations(self, ops_body: str) -> List[str]:
        """Extract operation types from the dispatch body"""
        operations = []
        
        # Common MLIR operations
        op_patterns = [
            r'linalg\.\w+',
            r'tensor\.\w+',
            r'arith\.\w+',
            r'math\.\w+',
            r'flow\.\w+',
            r'iree_tensor_ext\.\w+',
        ]
        
        for pattern in op_patterns:
            ops = re.findall(pattern, ops_body)
            operations.extend(ops)
        
        return list(set(operations))
    
    def _extract_tensors(self, body: str) -> Tuple[List[str], List[str]]:
        """Extract input and output tensor shapes"""
        inputs = []
        outputs = []
        
        # Extract tensor types
        tensor_pattern = r'tensor<([^>]+)>'
        tensors = re.findall(tensor_pattern, body)
        
        # Classify based on readonly/readwrite
        for match in re.finditer(r'(readonly|readwrite|writeonly):tensor<([^>]+)>', body):
            access = match.group(1)
            shape = match.group(2)
            
            if access == 'readonly':
                inputs.append(shape)
            else:
                outputs.append(shape)
        
        return inputs, outputs
    
    def parse_dependencies(self):
        """Parse the stream command execution to extract dispatch dependencies and concurrent groups"""
        # Find the stream execution command block
        cmd_execute_pattern = r'stream\.cmd\.execute[^{]*\{(.*?)\n\s*\}\s*=>\s*!stream\.timepoint'
        cmd_match = re.search(cmd_execute_pattern, self.content, re.DOTALL)
        
        if not cmd_match:
            print("Warning: Could not find stream.cmd.execute block. Dependencies may be incomplete.")
            self._infer_sequential_dependencies()
            return
        
        cmd_body = cmd_match.group(1)
        self._analyze_stream_commands(cmd_body)
        
        # Ensure all dispatches are accounted for in execution stages
        all_staged_dispatches = set()
        for stage in self.execution_stages:
            all_staged_dispatches.update(stage)
        
        missing_dispatches = set(self.dispatches.keys()) - all_staged_dispatches
        if missing_dispatches:
            print(f"Warning: {len(missing_dispatches)} dispatches not found in execution stages: {sorted(missing_dispatches)}")
            # Add them as individual sequential stages at the end
            for dispatch_id in sorted(missing_dispatches):
                self.execution_stages.append([dispatch_id])
                self.dispatches[dispatch_id].execution_order = len(self.execution_stages) - 1

        # Targeted debug for key dispatches around 9-12
        for target in [9, 10, 11, 12]:
            if target in self.dispatches:
                node = self.dispatches[target]
                print(
                    f"[DEBUG] dispatch_{target}: stage={node.execution_order}, "
                    f"deps={sorted(node.dependencies)}, concurrent_group={node.concurrent_group}"
                )
    
    def _analyze_stream_commands(self, cmd_body: str):
        """Analyze stream commands to determine execution order and concurrency"""
        execution_stage = 0
        previous_stage_dispatches = set()
        
        # Split by lines and track concurrent blocks
        lines = cmd_body.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for concurrent block
            if 'stream.cmd.concurrent' in line:
                # Extract the concurrent block
                concurrent_block, end_idx = self._extract_concurrent_block(lines, i)
                i = end_idx
                
                # Parse dispatches in concurrent block
                current_stage_dispatches = self._parse_dispatches_in_block(concurrent_block)
                
                if current_stage_dispatches:
                    print(f"[DEBUG] Concurrent stage {execution_stage}: dispatches {sorted(current_stage_dispatches)}; deps from prev stage {sorted(previous_stage_dispatches)}")
                    # All dispatches in concurrent block depend on previous stage
                    for dispatch_id in current_stage_dispatches:
                        if dispatch_id in self.dispatches:
                            self.dispatches[dispatch_id].concurrent_group = execution_stage
                            self.dispatches[dispatch_id].execution_order = execution_stage
                            self.dispatches[dispatch_id].dependencies.update(previous_stage_dispatches)
                    
                    self.concurrent_groups.append(current_stage_dispatches)
                    self.execution_stages.append(sorted(current_stage_dispatches))
                    previous_stage_dispatches = current_stage_dispatches
                    execution_stage += 1
            
            # Check for standalone dispatch (sequential)
            elif 'stream.cmd.dispatch' in line:
                dispatch_id = self._extract_dispatch_id(line)
                
                if dispatch_id is not None and dispatch_id in self.dispatches:
                    print(f"[DEBUG] Sequential stage {execution_stage}: dispatch {dispatch_id}; deps from prev stage {sorted(previous_stage_dispatches)}")
                    # Sequential dispatch depends on all previous stage dispatches
                    self.dispatches[dispatch_id].execution_order = execution_stage
                    self.dispatches[dispatch_id].dependencies.update(previous_stage_dispatches)
                    
                    self.execution_stages.append([dispatch_id])
                    previous_stage_dispatches = {dispatch_id}
                    execution_stage += 1
            
            i += 1
    
    def _extract_concurrent_block(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract a concurrent block and return its content and end index"""
        brace_count = 0
        block_lines = []
        found_start = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            if '{' in line:
                brace_count += line.count('{')
                found_start = True
            if '}' in line:
                brace_count -= line.count('}')
            
            if found_start:
                block_lines.append(line)
            
            if found_start and brace_count == 0:
                return '\n'.join(block_lines), i + 1
        
        return '\n'.join(block_lines), len(lines)
    
    def _parse_dispatches_in_block(self, block: str) -> Set[int]:
        """Extract all dispatch IDs from a block of code"""
        dispatch_ids = set()
        dispatch_pattern = r'@\w+\$async_dispatch_(\d+)::'
        
        for match in re.finditer(dispatch_pattern, block):
            dispatch_id = int(match.group(1))
            dispatch_ids.add(dispatch_id)
        
        return dispatch_ids
    
    def _extract_dispatch_id(self, line: str) -> Optional[int]:
        """Extract dispatch ID from a single line"""
        match = re.search(r'@\w+\$async_dispatch_(\d+)::', line)
        if match:
            return int(match.group(1))
        return None
    
    def _infer_sequential_dependencies(self):
        """Infer sequential dependencies as fallback"""
        sorted_ids = sorted(self.dispatches.keys())
        for i in range(1, len(sorted_ids)):
            dispatch_id = sorted_ids[i]
            prev_id = sorted_ids[i-1]
            self.dispatches[dispatch_id].dependencies.add(prev_id)
    
    def build_graph(self):
        """Build NetworkX DAG from parsed dispatches"""
        for dispatch_id, node in self.dispatches.items():
            # Add node with attributes
            self.graph.add_node(
                dispatch_id,
                name=node.name,
                export=node.workgroup_export,
                operations=node.operations,
                inputs=node.input_tensors,
                outputs=node.output_tensors,
                label=f"dispatch_{dispatch_id}",
                concurrent_group=node.concurrent_group,
                execution_order=node.execution_order
            )
            
            # Add edges for dependencies
            for dep_id in node.dependencies:
                if dep_id in self.dispatches:
                    self.graph.add_edge(dep_id, dispatch_id)
    
    def visualize(self, output_path: str = None):
        """Visualize the dependency graph with concurrent groups highlighted"""
        plt.figure(figsize=(20, 14))
        
        # Use hierarchical layout for DAG
        # Group nodes by execution order
        pos = {}
        y_spacing = 2.0
        max_nodes_per_stage = max(len(stage) for stage in self.execution_stages) if self.execution_stages else 1
        x_spacing = 10.0 / max(max_nodes_per_stage, 1)
        
        for stage_idx, stage in enumerate(self.execution_stages):
            y = -stage_idx * y_spacing
            num_nodes = len(stage)
            start_x = -(num_nodes - 1) * x_spacing / 2
            
            for node_idx, dispatch_id in enumerate(stage):
                x = start_x + node_idx * x_spacing
                pos[dispatch_id] = (x, y)
        
        # Handle any nodes not in execution_stages (orphaned nodes)
        missing_nodes = set(self.graph.nodes()) - set(pos.keys())
        if missing_nodes:
            print(f"Warning: {len(missing_nodes)} nodes not in execution stages: {sorted(missing_nodes)}")
            # Add them at the bottom
            y = -len(self.execution_stages) * y_spacing if self.execution_stages else 0
            for idx, node in enumerate(sorted(missing_nodes)):
                pos[node] = (idx * x_spacing, y - y_spacing)
        
        # Color nodes by concurrent group and operation type
        node_colors = []
        for node in self.graph.nodes():
            ops = self.graph.nodes[node].get('operations', [])
            concurrent_group = self.graph.nodes[node].get('concurrent_group')
            
            if any('conv' in op.lower() for op in ops):
                node_colors.append('lightblue')
            elif any('matmul' in op.lower() or 'matvec' in op.lower() for op in ops):
                node_colors.append('lightcoral')
            elif any('memcpy' in op or 'copy' in op for op in ops):
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        # Draw graph
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=2500,
            with_labels=True,
            labels={n: f"D{n}" for n in self.graph.nodes()},
            font_size=9,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            arrowstyle='->',
            node_shape='o'
        )
        
        # Draw rectangles around concurrent groups
        if self.execution_stages:
            for stage_idx, stage in enumerate(self.execution_stages):
                if len(stage) > 1:  # Only highlight concurrent groups
                    y = -stage_idx * y_spacing
                    num_nodes = len(stage)
                    start_x = -(num_nodes - 1) * x_spacing / 2
                    end_x = start_x + (num_nodes - 1) * x_spacing
                    
                    # Draw a rectangle around concurrent dispatches
                    rect = plt.Rectangle(
                        (start_x - 0.8, y - 0.5), 
                        end_x - start_x + 1.6, 
                        1.0,
                        fill=False, 
                        edgecolor='red', 
                        linewidth=2, 
                        linestyle='--',
                        label=f'Concurrent Group {stage_idx}' if stage_idx == 0 else ''
                    )
                    plt.gca().add_patch(rect)
        
        plt.title(f"MLIR Stream Dispatch Dependency Graph\n{self.mlir_file_path.name}\n(Red dashed boxes = concurrent execution)", 
                  fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_to_json(self, output_path: str):
        """Export graph structure to JSON with concurrent execution info"""
        data = {
            'source_file': str(self.mlir_file_path),
            'num_dispatches': len(self.dispatches),
            'num_execution_stages': len(self.execution_stages),
            'num_concurrent_groups': len([s for s in self.execution_stages if len(s) > 1]),
            'dispatches': {},
            'edges': [],
            'execution_stages': [],
            'concurrent_groups': []
        }
        
        for dispatch_id, node in self.dispatches.items():
            data['dispatches'][f'dispatch_{dispatch_id}'] = {
                'id': dispatch_id,
                'name': node.name,
                'export': node.workgroup_export,
                'operations': node.operations,
                'input_tensors': node.input_tensors,
                'output_tensors': node.output_tensors,
                'dependencies': list(node.dependencies),
                'concurrent_group': node.concurrent_group,
                'execution_order': node.execution_order,
                'metadata': node.metadata
            }
        
        for edge in self.graph.edges():
            data['edges'].append({
                'from': f'dispatch_{edge[0]}',
                'to': f'dispatch_{edge[1]}'
            })
        
        # Add execution stages info
        for stage_idx, stage in enumerate(self.execution_stages):
            stage_info = {
                'stage': stage_idx,
                'dispatches': [f'dispatch_{d}' for d in stage],
                'is_concurrent': len(stage) > 1,
                'num_dispatches': len(stage)
            }
            data['execution_stages'].append(stage_info)
        
        # Add concurrent groups
        for group_idx, group in enumerate(self.concurrent_groups):
            data['concurrent_groups'].append({
                'group': group_idx,
                'dispatches': [f'dispatch_{d}' for d in sorted(group)]
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Graph data exported to: {output_path}")
    
    def get_critical_path(self) -> List[int]:
        """Get the critical path (longest path) in the DAG"""
        if not nx.is_directed_acyclic_graph(self.graph):
            print("Warning: Graph contains cycles!")
            return []
        
        try:
            path = nx.dag_longest_path(self.graph)
            return path
        except:
            return []
    
    def get_parallel_stages(self) -> List[Set[int]]:
        """Get dispatch stages that can be executed in parallel"""
        return [set(stage) for stage in self.execution_stages if len(stage) > 1]
    
    def get_concurrent_groups(self) -> List[Set[int]]:
        """Get all concurrent groups of dispatches"""
        return self.concurrent_groups
    
    def get_max_parallelism(self) -> int:
        """Get maximum number of dispatches that can run in parallel"""
        if not self.execution_stages:
            return 0
        return max(len(stage) for stage in self.execution_stages)
    
    def print_summary(self):
        """Print summary of parsed information"""
        print(f"\n{'='*70}")
        print(f"MLIR Stream Analysis: {self.mlir_file_path.name}")
        print(f"{'='*70}")
        print(f"Total Dispatches: {len(self.dispatches)}")
        print(f"Total Dependencies (edges): {self.graph.number_of_edges()}")
        print(f"Execution Stages: {len(self.execution_stages)}")
        print(f"Concurrent Groups: {len(self.concurrent_groups)}")
        print(f"Max Parallelism: {self.get_max_parallelism()} dispatches")
        print(f"\nDispatches:")
        print(f"{'-'*70}")
        
        for dispatch_id in sorted(self.dispatches.keys()):
            node = self.dispatches[dispatch_id]
            deps_str = ', '.join(map(str, sorted(node.dependencies))) if node.dependencies else 'None'
            
            # Get concurrent peers (other dispatches in same stage)
            concurrent_peers = []
            if node.execution_order < len(self.execution_stages):
                stage = self.execution_stages[node.execution_order]
                concurrent_peers = [d for d in stage if d != dispatch_id]
            
            print(f"  dispatch_{dispatch_id}:")
            print(f"    Name: {node.name}")
            print(f"    Export: {node.workgroup_export}")
            print(f"    Execution Stage: {node.execution_order}")
            if concurrent_peers:
                print(f"    Concurrent With: [{', '.join(map(str, concurrent_peers))}]")
            print(f"    Dependencies: [{deps_str}]")
            if node.operations:
                ops_summary = ', '.join(sorted(set(node.operations))[:3])
                if len(node.operations) > 3:
                    ops_summary += f", ... ({len(set(node.operations))} total)"
                print(f"    Operations: {ops_summary}")
            if node.input_tensors:
                print(f"    Inputs: {len(node.input_tensors)} tensors")
            if node.output_tensors:
                print(f"    Outputs: {len(node.output_tensors)} tensors")
            print()
        
        # Critical path
        critical_path = self.get_critical_path()
        if critical_path:
            print(f"Critical Path: {' -> '.join(map(str, critical_path))}")
            print(f"Critical Path Length: {len(critical_path)}")
        
        # Execution stages
        print(f"\nExecution Stages (Sequential Order):")
        print(f"{'-'*70}")
        for stage_idx, stage in enumerate(self.execution_stages):
            concurrent_marker = " [CONCURRENT]" if len(stage) > 1 else ""
            print(f"  Stage {stage_idx}{concurrent_marker}: {sorted(stage)}")
        
        # Parallel stages summary
        parallel_stages = self.get_parallel_stages()
        if parallel_stages:
            print(f"\nConcurrent Execution Opportunities:")
            print(f"{'-'*70}")
            for i, stage in enumerate(parallel_stages):
                print(f"  Concurrent Group {i}: {sorted(stage)} ({len(stage)} dispatches)")
        
        print(f"{'='*70}\n")
    
    def parse(self):
        """Main parsing pipeline"""
        print(f"Parsing MLIR file: {self.mlir_file_path}")
        self.load_file()
        self.parse_dispatches()
        self.parse_dependencies()
        self.build_graph()
        print("Parsing complete!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse MLIR stream files and extract dispatch dependencies with concurrency info'
    )
    parser.add_argument(
        'mlir_file',
        type=str,
        help='Path to MLIR stream file'
    )
    parser.add_argument(
        '--visualize',
        '-v',
        type=str,
        help='Output path for visualization (PNG)',
        default=None
    )
    parser.add_argument(
        '--export-json',
        '-j',
        type=str,
        help='Export graph to JSON file',
        default=None
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing summary'
    )
    
    args = parser.parse_args()
    
    # Parse MLIR file
    mlir_parser = MLIRStreamParser(args.mlir_file)
    mlir_parser.parse()
    
    # Print summary
    if not args.no_summary:
        mlir_parser.print_summary()
    
    # Visualize
    if args.visualize:
        mlir_parser.visualize(args.visualize)
    
    # Export to JSON
    if args.export_json:
        mlir_parser.export_to_json(args.export_json)
    
    # Return graph for programmatic use
    return mlir_parser.graph, mlir_parser.dispatches


if __name__ == '__main__':
    main()
