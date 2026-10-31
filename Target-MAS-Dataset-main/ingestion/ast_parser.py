"""
TraceProfiler AST Parser Module.

Provides framework-independent Python AST parsing, symbol resolution,
scope tracking, source hashing, and static evaluation without executing target code.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SourceLocation:
    file_path: str
    start_line: int
    end_line: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class ImportItem:
    name: str
    asname: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.asname if self.asname else self.name


@dataclass
class ImportReference:
    module: Optional[str]
    names: List[ImportItem]
    is_from: bool
    level: int = 0
    start_line: int = 0
    end_line: int = 0


@dataclass
class CallInfo:
    func_name: str
    caller_obj: Optional[str]
    method_name: Optional[str]
    args: List[Any]
    keywords: Dict[str, Any]
    raw_node: Optional[ast.Call] = None
    start_line: int = 0
    end_line: int = 0
    assigned_to: Optional[str] = None


@dataclass
class AssignmentInfo:
    targets: List[str]
    value_repr: str
    value_node: Optional[ast.AST] = None
    call_info: Optional[CallInfo] = None
    start_line: int = 0
    end_line: int = 0


@dataclass
class FunctionDefInfo:
    name: str
    args: List[str]
    keyword_args: Dict[str, Any]
    decorators: List[str]
    docstring: Optional[str]
    return_annotation: Optional[str]
    start_line: int
    end_line: int
    is_method: bool = False
    parent_class: Optional[str] = None
    calls: List[CallInfo] = field(default_factory=list)
    assignments: List[AssignmentInfo] = field(default_factory=list)
    returns: List[Any] = field(default_factory=list)
    local_variables: Set[str] = field(default_factory=set)


@dataclass
class ClassDefInfo:
    name: str
    bases: List[str]
    decorators: List[str]
    docstring: Optional[str]
    start_line: int
    end_line: int
    methods: Dict[str, FunctionDefInfo] = field(default_factory=dict)
    class_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedModule:
    file_path: str
    source_code: str
    sha256: str
    ast_tree: Optional[ast.AST]
    parse_status: str  # "parsed" | "failed"
    error_message: Optional[str] = None
    imports: List[ImportReference] = field(default_factory=list)
    functions: Dict[str, FunctionDefInfo] = field(default_factory=dict)
    classes: Dict[str, ClassDefInfo] = field(default_factory=dict)
    top_level_calls: List[CallInfo] = field(default_factory=list)
    top_level_assignments: List[AssignmentInfo] = field(default_factory=list)
    all_calls: List[CallInfo] = field(default_factory=list)
    symbol_table: Dict[str, str] = field(default_factory=dict)  # name -> source/type
    local_variables: Set[str] = field(default_factory=set)

    def is_local_symbol(self, name: str) -> bool:
        """Check if symbol is defined in module (function, class, variable, or imported)."""
        return (
            name in self.functions
            or name in self.classes
            or name in self.local_variables
            or name in self.symbol_table
        )

    def resolve_symbol_origin(self, name: str) -> Optional[Dict[str, Any]]:
        """Resolve symbol to import origin or local declaration."""
        if name in self.symbol_table:
            return {"type": "symbol_table", "origin": self.symbol_table[name]}
        if name in self.functions:
            fn = self.functions[name]
            return {"type": "local_function", "name": name, "start_line": fn.start_line, "end_line": fn.end_line}
        if name in self.classes:
            cls_def = self.classes[name]
            return {"type": "local_class", "name": name, "start_line": cls_def.start_line, "end_line": cls_def.end_line}
        if name in self.local_variables:
            return {"type": "local_variable", "name": name}
        return None

    def get_source_segment(self, start_line: int, end_line: int) -> str:
        """Extract lines from source safely."""
        lines = self.source_code.splitlines()
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)
        return "\n".join(lines[start:end])


# ============================================================================
# Static Evaluation Helpers
# ============================================================================

def safe_eval_literal(node: Optional[ast.AST]) -> Tuple[bool, Any, str]:
    """
    Safely evaluate a literal AST node without using eval() or exec().

    Returns:
        (is_static, value, string_representation)
    """
    if node is None:
        return True, None, "None"

    # Python 3.8+ Constant
    if isinstance(node, ast.Constant):
        return True, node.value, repr(node.value)

    # Legacy Python <3.8 fallback if attributes exist
    legacy_str = getattr(ast, "Str", None)
    legacy_bytes = getattr(ast, "Bytes", None)
    legacy_num = getattr(ast, "Num", None)
    legacy_nc = getattr(ast, "NameConstant", None)

    if legacy_str and isinstance(node, legacy_str):
        return True, node.s, repr(node.s)
    if legacy_bytes and isinstance(node, legacy_bytes):
        return True, node.s, repr(node.s)
    if legacy_num and isinstance(node, legacy_num):
        return True, node.n, repr(node.n)
    if legacy_nc and isinstance(node, legacy_nc):
        return True, node.value, repr(node.value)

    # List
    if isinstance(node, ast.List):
        items = []
        is_all_static = True
        for elt in node.elts:
            stat, val, _ = safe_eval_literal(elt)
            if not stat:
                is_all_static = False
            items.append(val if stat else ast.unparse(elt))
        return is_all_static, items, f"[{', '.join(repr(x) for x in items)}]"

    # Tuple
    if isinstance(node, ast.Tuple):
        items = []
        is_all_static = True
        for elt in node.elts:
            stat, val, _ = safe_eval_literal(elt)
            if not stat:
                is_all_static = False
            items.append(val if stat else ast.unparse(elt))
        return is_all_static, tuple(items), f"({', '.join(repr(x) for x in items)})"

    # Dict
    if isinstance(node, ast.Dict):
        result_dict = {}
        is_all_static = True
        for k_node, v_node in zip(node.keys, node.values):
            if k_node is not None:
                k_stat, k_val, _ = safe_eval_literal(k_node)
                v_stat, v_val, _ = safe_eval_literal(v_node)
                if not (k_stat and v_stat):
                    is_all_static = False
                key_key = k_val if k_stat else ast.unparse(k_node)
                val_val = v_val if v_stat else ast.unparse(v_node)
                result_dict[key_key] = val_val
            else:
                is_all_static = False
        return is_all_static, result_dict, repr(result_dict)

    # String Concatenation: "a" + "b"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left_stat, left_val, _ = safe_eval_literal(node.left)
        right_stat, right_val, _ = safe_eval_literal(node.right)
        if left_stat and right_stat and isinstance(left_val, str) and isinstance(right_val, str):
            res = left_val + right_val
            return True, res, repr(res)

    # UnaryOp (e.g. -1, +5)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            sub_stat, sub_val, _ = safe_eval_literal(node.operand)
            if sub_stat and isinstance(sub_val, (int, float)):
                return True, -sub_val, str(-sub_val)
        if isinstance(node.op, ast.UAdd):
            add_stat, add_val, _ = safe_eval_literal(node.operand)
            if add_stat and isinstance(add_val, (int, float)):
                return True, add_val, str(add_val)

    # Identifiers / Names (e.g. True, False, None if represented as Name)
    if isinstance(node, ast.Name):
        if node.id == "None":
            return True, None, "None"
        if node.id == "True":
            return True, True, "True"
        if node.id == "False":
            return True, False, "False"
        return False, None, node.id

    # Dynamic or complex expressions
    try:
        raw_repr = ast.unparse(node)
    except Exception:
        raw_repr = "<unparseable_node>"
    return False, None, raw_repr


def extract_call_name(node: ast.AST) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract (full_func_name, caller_obj, method_name) from Call.func.
    e.g.
    `intake_agent.create_agent` -> ("intake_agent.create_agent", "intake_agent", "create_agent")
    `create_agent` -> ("create_agent", None, "create_agent")
    `crew.kickoff` -> ("crew.kickoff", "crew", "kickoff")
    """
    if isinstance(node, ast.Name):
        return node.id, None, node.id
    if isinstance(node, ast.Attribute):
        caller_str = ""
        try:
            caller_str = ast.unparse(node.value)
        except Exception:
            caller_str = getattr(node.value, "id", "<unknown>")
        full_name = f"{caller_str}.{node.attr}"
        return full_name, caller_str, node.attr
    try:
        raw = ast.unparse(node)
        return raw, None, raw
    except Exception:
        return "<complex_call>", None, "<complex_call>"


def extract_decorator_names(decorator_list: List[ast.AST]) -> List[str]:
    """Extract decorator names or calls."""
    results = []
    for dec in decorator_list:
        if isinstance(dec, ast.Name):
            results.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            results.append(ast.unparse(dec))
        elif isinstance(dec, ast.Call):
            try:
                results.append(ast.unparse(dec.func))
            except Exception:
                results.append("<decorator_call>")
        else:
            try:
                results.append(ast.unparse(dec))
            except Exception:
                pass
    return results


# ============================================================================
# AST Visitor / Extraction Engine
# ============================================================================

class ModuleASTVisitor(ast.NodeVisitor):
    """Visits a module's AST to extract all structural entities, calls, scopes, and statements."""

    def __init__(self, file_path: str, source_code: str) -> None:
        self.file_path = file_path
        self.source_code = source_code
        self.imports: List[ImportReference] = []
        self.functions: Dict[str, FunctionDefInfo] = {}
        self.classes: Dict[str, ClassDefInfo] = {}
        self.top_level_calls: List[CallInfo] = []
        self.top_level_assignments: List[AssignmentInfo] = []
        self.all_calls: List[CallInfo] = []
        self.symbol_table: Dict[str, str] = {}
        self.local_variables: Set[str] = set()

        self._current_function: Optional[FunctionDefInfo] = None
        self._current_class: Optional[ClassDefInfo] = None

    def visit_Import(self, node: ast.Import) -> None:
        items = [ImportItem(name=alias.name, asname=alias.asname) for alias in node.names]
        ref = ImportReference(
            module=None,
            names=items,
            is_from=False,
            level=0,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )
        self.imports.append(ref)
        for item in items:
            eff_name = item.effective_name
            self.symbol_table[eff_name] = f"import:{item.name}"
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module
        items = [ImportItem(name=alias.name, asname=alias.asname) for alias in node.names]
        ref = ImportReference(
            module=mod,
            names=items,
            is_from=True,
            level=node.level,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )
        self.imports.append(ref)
        for item in items:
            eff_name = item.effective_name
            full_mod = mod or ""
            self.symbol_table[eff_name] = f"from:{full_mod}:{item.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                bases.append(getattr(b, "id", "<base>"))

        decs = extract_decorator_names(node.decorator_list)
        docstring = ast.get_docstring(node)

        cls_info = ClassDefInfo(
            name=node.name,
            bases=bases,
            decorators=decs,
            docstring=docstring,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )
        self.classes[node.name] = cls_info
        self.symbol_table[node.name] = f"class:{node.name}"

        prev_class = self._current_class
        self._current_class = cls_info
        self.generic_visit(node)
        self._current_class = prev_class

    def visit_FunctionDef(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        args = [a.arg for a in node.args.args]
        kw_args: Dict[str, Any] = {}
        # match defaults from right
        defaults = node.args.defaults
        offset = len(args) - len(defaults)
        for idx, def_node in enumerate(defaults):
            arg_name = args[offset + idx]
            _, val, repr_v = safe_eval_literal(def_node)
            kw_args[arg_name] = val if val is not None else repr_v

        decs = extract_decorator_names(node.decorator_list)
        docstring = ast.get_docstring(node)
        ret_ann = None
        if node.returns:
            try:
                ret_ann = ast.unparse(node.returns)
            except Exception:
                ret_ann = "<ann>"

        is_method = self._current_class is not None
        parent_class_name = self._current_class.name if self._current_class else None

        fn_info = FunctionDefInfo(
            name=node.name,
            args=args,
            keyword_args=kw_args,
            decorators=decs,
            docstring=docstring,
            return_annotation=ret_ann,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            is_method=is_method,
            parent_class=parent_class_name,
        )

        if is_method and self._current_class:
            self._current_class.methods[node.name] = fn_info
        else:
            self.functions[node.name] = fn_info
            self.symbol_table[node.name] = f"function:{node.name}"

        prev_func = self._current_function
        self._current_function = fn_info
        self.generic_visit(node)
        self._current_function = prev_func

    def visit_Call(self, node: ast.Call) -> None:
        full_name, caller, method = extract_call_name(node.func)

        # Positional arguments
        args = []
        for arg in node.args:
            stat, val, raw_repr = safe_eval_literal(arg)
            args.append(val if stat else raw_repr)

        # Keyword arguments
        keywords = {}
        for kw in node.keywords:
            if kw.arg:
                stat, val, raw_repr = safe_eval_literal(kw.value)
                keywords[kw.arg] = val if val is not None else raw_repr

        call_info = CallInfo(
            func_name=full_name,
            caller_obj=caller,
            method_name=method,
            args=args,
            keywords=keywords,
            raw_node=node,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )

        self.all_calls.append(call_info)

        if self._current_function:
            self._current_function.calls.append(call_info)
        elif not self._current_class:
            self.top_level_calls.append(call_info)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = []
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                targets.append(tgt.id)
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                for elt in tgt.elts:
                    if isinstance(elt, ast.Name):
                        targets.append(elt.id)
            else:
                try:
                    targets.append(ast.unparse(tgt))
                except Exception:
                    pass

        val_repr = ""
        try:
            val_repr = ast.unparse(node.value)
        except Exception:
            val_repr = "<expr>"

        # Check if right-hand side is a Call
        call_info = None
        if isinstance(node.value, ast.Call):
            full_name, caller, method = extract_call_name(node.value.func)
            args = []
            for arg in node.value.args:
                stat, val, r = safe_eval_literal(arg)
                args.append(val if stat else r)
            kws = {}
            for kw in node.value.keywords:
                if kw.arg:
                    stat, val, r = safe_eval_literal(kw.value)
                    kws[kw.arg] = val if stat else r
            call_info = CallInfo(
                func_name=full_name,
                caller_obj=caller,
                method_name=method,
                args=args,
                keywords=kws,
                raw_node=node.value,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                assigned_to=targets[0] if targets else None,
            )

        assign_info = AssignmentInfo(
            targets=targets,
            value_repr=val_repr,
            value_node=node.value,
            call_info=call_info,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )

        for t in targets:
            if self._current_function:
                self._current_function.local_variables.add(t)
            else:
                self.local_variables.add(t)
                self.symbol_table[t] = f"variable:{t}"

        if self._current_function:
            self._current_function.assignments.append(assign_info)
        elif not self._current_class:
            self.top_level_assignments.append(assign_info)

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self._current_function and node.value:
            stat, val, raw_repr = safe_eval_literal(node.value)
            self._current_function.returns.append({
                "static": stat,
                "value": val if stat else None,
                "representation": raw_repr,
                "node": node.value,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
            })
        self.generic_visit(node)


# ============================================================================
# Main Parser Interface & Caching
# ============================================================================

class ASTParser:
    """Parser cache and entrypoint for safe Python AST inspection."""

    def __init__(self) -> None:
        self.cache: Dict[str, ParsedModule] = {}

    def parse_source(self, source_code: str, file_path: str = "<memory>") -> ParsedModule:
        """Parse in-memory source code safely."""
        sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest()

        try:
            tree = ast.parse(source_code, filename=file_path)
            visitor = ModuleASTVisitor(file_path=file_path, source_code=source_code)
            visitor.visit(tree)

            parsed = ParsedModule(
                file_path=file_path,
                source_code=source_code,
                sha256=sha256,
                ast_tree=tree,
                parse_status="parsed",
                error_message=None,
                imports=visitor.imports,
                functions={fn.name: fn for fn in visitor.functions} if isinstance(visitor.functions, list) else visitor.functions,
                classes=visitor.classes,
                top_level_calls=visitor.top_level_calls,
                top_level_assignments=visitor.top_level_assignments,
                all_calls=visitor.all_calls,
                symbol_table=visitor.symbol_table,
                local_variables=visitor.local_variables,
            )
            return parsed
        except SyntaxError as e:
            return ParsedModule(
                file_path=file_path,
                source_code=source_code,
                sha256=sha256,
                ast_tree=None,
                parse_status="failed",
                error_message=f"SyntaxError: {str(e)} at line {e.lineno}",
            )
        except Exception as e:
            return ParsedModule(
                file_path=file_path,
                source_code=source_code,
                sha256=sha256,
                ast_tree=None,
                parse_status="failed",
                error_message=f"ParseException: {str(e)}",
            )

    def parse_file(self, file_path: Union[str, Path]) -> ParsedModule:
        """Read and parse a file from disk, using cached representation if available."""
        abs_path = str(Path(file_path).resolve())
        if abs_path in self.cache:
            return self.cache[abs_path]

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                source = f.read()
            parsed = self.parse_source(source, file_path=abs_path)
        except Exception as e:
            parsed = ParsedModule(
                file_path=abs_path,
                source_code="",
                sha256="",
                ast_tree=None,
                parse_status="failed",
                error_message=f"FileReadError: {str(e)}",
            )

        self.cache[abs_path] = parsed
        return parsed

    def clear_cache(self) -> None:
        self.cache.clear()


# Global default instance
default_ast_parser = ASTParser()
parse_file = default_ast_parser.parse_file
parse_source = default_ast_parser.parse_source
