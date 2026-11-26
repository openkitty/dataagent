"""Offline acceptance for migrated history corpus preparation and native task routing."""

import hashlib
import json
import stat
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from conftest import ScriptedModel, call
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dataagent import LaunchOptions, create_agent, prepare_runtime
from dataagent.cli import main as dataagent_main
from dataagent.config.declarations import SubAgentSpec
from dataagent.config.strict_yaml import read_yaml
from dataagent.diagnostics.errors import DataAgentError
from dataagent.extensions.loading import PythonLoader
from dataagent.semantic_client import SemanticServiceClient

PLUGIN = Path(__file__).resolve().parents[1] / "dataagent/plugins/de-agent"

cli = import_module("dataagent.plugins.de-agent.scripts.history_sql")
downloader = import_module("dataagent.plugins.de-agent.utils.history_sql.downloader")
corpus_path = cli.corpus_path
CorpusDownloadError = downloader.CorpusDownloadError
compare_corpora = downloader.compare_corpora
fetch_all_entities = downloader.fetch_all_entities
write_corpus = downloader.write_corpus
_unwrap_jsonb = downloader._unwrap_jsonb
_is_verified_sql = downloader._is_verified_sql
_verified_state = downloader._verified_state


def entity(guid="one", table="db.demo", sql="SELECT 1", **attributes):
    return {"guid": guid, "attributes": {"target_table": table, "expression": sql, **attributes}}


def hooks():
    return PythonLoader().load(PLUGIN, "hooks/history_sql.py:prepare_search")


def request(context, name="task", **args):
    return ToolCallRequest(
        tool_call={"name": name, "id": "lookup", "args": args},
        tool=None,
        state={"messages": [HumanMessage(content="original requirement")]},
        runtime=SimpleNamespace(context=context),
    )


def context(tmp_path, **options):
    root = tmp_path / "corpus"
    root.mkdir(exist_ok=True)
    (root / "DEMO.sql").write_text(
        "INSERT OVERWRITE db.demo SELECT user_id FROM db.download GROUP BY user_id"
    )
    config = {"history_sql": {"enabled": True, "corpus_path": str(root), **options}}
    return SimpleNamespace(
        home=tmp_path,
        session=SimpleNamespace(outputs=tmp_path / "outputs"),
        get_plugin_config=lambda _: config,
        semantic_client=Mock(side_effect=AssertionError("History must not call semantic APIs")),
    )


def test_paging_uses_get_basic_offsets_and_preserves_distinct_records():
    seen = []

    def respond(req):
        assert req.method == "GET"
        assert req.url.path.endswith("/search/basic")
        assert req.url.params["typeName"] == "sql_process"
        offset = int(req.url.params["offset"])
        seen.append(offset)
        return httpx.Response(
            200,
            json={
                "entities": {
                    0: [entity("one"), entity("two")],
                    2: [entity("two"), entity("three")],
                    4: [],
                }[offset]
            },
        )

    with SemanticServiceClient("http://semantic", transport=httpx.MockTransport(respond)) as client:
        result = fetch_all_entities(client, page_size=2)
    assert seen == [0, 2, 4]
    assert [item["guid"] for item in result.entities] == ["one", "two", "three"]
    assert result.duplicates_skipped == 1


def test_paging_stagnation_fails_instead_of_hanging():
    client = SimpleNamespace(search_entities=lambda *_args, **_kw: [entity()])
    with pytest.raises(CorpusDownloadError):
        fetch_all_entities(client, max_stagnant_pages=2)


@pytest.mark.parametrize("payload", [{}, {"entities": {}}, {"entities": ["bad"]}, "bad"])
def test_malformed_entity_pages_are_not_reported_as_empty(payload):
    with (
        SemanticServiceClient(
            "http://semantic",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
        ) as client,
        pytest.raises((ValueError, DataAgentError)),
    ):
        client.search_entities("sql_process")


def test_corpus_preserves_duplicates_manifest_and_unmanaged_files(tmp_path):
    root = tmp_path / "corpus"
    rows = [entity("one", sql="SELECT 1"), entity("two", sql="SELECT 2")]
    stats = write_corpus(rows, root, lock=False)
    assert stats.written == stats.group_suffixed == 2
    files = sorted(root.glob("*.sql"))
    assert len(files) == 2 and not (root / "DEMO.sql").exists()
    manifest = json.loads((root / ".history_sql_manifest.json").read_text())
    for item in manifest["files"]:
        assert item["sha256"] == hashlib.sha256((root / item["file"]).read_bytes()).hexdigest()
    (root / "MANUAL.sql").write_text("my SQL")
    files[0].write_text("user modified")
    stats = write_corpus([], root, lock=False)
    assert stats.pruned == [files[1].name]
    assert files[0].read_text() == "user modified"
    assert (root / "MANUAL.sql").read_text() == "my SQL"


def test_corpus_does_not_overwrite_foreign_sql(tmp_path):
    (tmp_path / "DEMO.sql").write_text("manual")
    with pytest.raises(CorpusDownloadError, match="unmanaged"):
        write_corpus([entity()], tmp_path, lock=False)
    assert (tmp_path / "DEMO.sql").read_text() == "manual"


def test_corpus_fallback_deleted_empty_and_read_only_snapshot(tmp_path):
    root = tmp_path / "corpus"
    deleted = {**entity("deleted"), "status": "DELETED"}
    stats = write_corpus(
        [
            entity("fallback", table=None, sql="INSERT OVERWRITE db.other SELECT 1"),
            entity("directory", table="DIRECTORY /tmp", sql_id="DUMP_DATA_123456789abc"),
            entity("empty", sql=""),
            deleted,
        ],
        root,
    )
    try:
        assert stats.written == stats.fallback_named == 2
        assert (root / "OTHER.sql").exists() and (root / "DUMP_DATA.sql").exists()
        assert stat.S_IMODE((root / "OTHER.sql").stat().st_mode) == 0o444
        assert len(stats.skipped) == 2
        assert compare_corpora(root, root).shared_content_diff == []
    finally:
        root.chmod(0o755)
        for path in root.iterdir():
            path.chmod(0o644)


def test_snapshot_path_priority(tmp_path, monkeypatch):
    monkeypatch.delenv("DATAAGENT_HISTORY_SQL_CORPUS", raising=False)
    assert corpus_path({}, tmp_path) == tmp_path / "history_sql"
    assert (
        corpus_path({"corpus_path": str(tmp_path / "configured")}, tmp_path)
        == tmp_path / "configured"
    )
    monkeypatch.setenv("DATAAGENT_HISTORY_SQL_CORPUS", str(tmp_path / "pinned"))
    assert corpus_path({"corpus_path": "ignored"}, tmp_path) == tmp_path / "pinned"


@pytest.mark.parametrize("status,expected_calls", [(500, 3), (401, 1)])
def test_downloader_retries_server_failures_but_not_authentication(
    status, expected_calls, monkeypatch
):
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    calls = []

    def respond(req):
        calls.append(req)
        return httpx.Response(status, json={"error": "unavailable"})

    with SemanticServiceClient("http://semantic", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(CorpusDownloadError):
            fetch_all_entities(client)
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "options,expected",
    [({"enabled": False}, "disabled"), ({"corpus_path": "/missing-history-corpus"}, "not_ready")],
)
def test_disabled_or_unprepared_search_does_not_start_child(tmp_path, options, expected):
    ctx = context(tmp_path, **options)
    result = hooks()(
        request(ctx, subagent_type="history-sql-recall", description="下载 / user_id"), params={}
    )
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert json.loads(result.content)["status"] == expected


def test_search_injects_corpus_without_semantic_preload(tmp_path):
    ctx = context(tmp_path)
    req = request(ctx, subagent_type="history-sql-recall", description="下载次数 / user_id")
    updated = hooks()(req, params={})
    assert updated.tool_call["id"] == req.tool_call["id"]
    assert str(tmp_path / "corpus") in updated.tool_call["args"]["description"]
    metadata = PythonLoader().load(PLUGIN, "hooks/metadata.py:prepare_recall")
    assert metadata(req, params={}) is None
    ctx.semantic_client.assert_not_called()
    assert not ctx.session.outputs.exists()


@pytest.mark.parametrize(
    "name,args",
    [
        ("execute", {"command": "cat /etc/passwd"}),
        ("read_file", {"file_path": "/etc/passwd"}),
        ("grep", {"pattern": "id"}),
        ("glob", {"pattern": "../*.sql"}),
    ],
)
def test_child_rejects_non_corpus_paths_and_write_execution_tools(tmp_path, name, args):
    ctx = context(tmp_path)
    if name == "glob":
        args["path"] = str(tmp_path / "corpus")
    guard = PythonLoader().load(PLUGIN, "hooks/history_sql.py:guard_files")
    assert guard(request(ctx, name=name, **args), params={}).status == "error"


def test_child_can_read_native_spilled_results_but_not_unrelated_artifacts(tmp_path):
    ctx = context(tmp_path)
    large = ctx.session.outputs / "artifacts/large_tool_results/search"
    large.parent.mkdir(parents=True)
    large.write_text("long grep results")
    other = ctx.session.outputs / "artifacts/private.txt"
    other.write_text("not a search result")
    guard = PythonLoader().load(PLUGIN, "hooks/history_sql.py:guard_files")
    assert guard(request(ctx, name="read_file", file_path=str(large)), params={}) is None
    assert guard(request(ctx, name="read_file", file_path=str(other)), params={}).status == "error"


def test_history_reference_is_explicit_bounded_and_excluded_from_ir(tmp_path):
    ctx = context(tmp_path)
    ctx.session.outputs.mkdir()
    loader = PythonLoader()
    reference = loader.load(PLUGIN, "hooks/history_sql.py:reference_context")
    description = json.dumps({"history_sql_reference": "来源 DEMO.sql：使用 bicoredata.SHA256"})
    result = reference(ctx, description)
    assert result["trust"] == "unverified_reference"
    assert reference(ctx, "{}") is None  # no automatic cross-task inheritance
    for value in ["", "x" * 2001, {}]:
        with pytest.raises(ValueError):
            reference(ctx, json.dumps({"history_sql_reference": value}))
    ctx.get_plugin_config = lambda _: {"history_sql": {"enabled": False}}
    with pytest.raises(ValueError, match="disabled"):
        reference(ctx, description)
    collect = loader.load(PLUGIN, "utils/data_task_ir/inputs.py:collect_evidence")
    assert (
        collect(
            ctx,
            {
                "messages": [
                    call(
                        "task",
                        {"subagent_type": "history-sql-recall", "description": "lookup"},
                        "lookup",
                    ),
                    ToolMessage(content="history SQL says deduplicate", tool_call_id="lookup"),
                ]
            },
        )
        == []
    )


def test_cli_disabled_skips_client_and_corpus_creation(tmp_path, monkeypatch, capsys):
    rt = SimpleNamespace(
        settings=SimpleNamespace(plugins=SimpleNamespace(config={}), semantic=None),
        paths=SimpleNamespace(home=tmp_path),
    )
    monkeypatch.setattr(cli, "prepare_runtime", lambda _: rt)
    factory = Mock(side_effect=AssertionError("must not create client"))
    monkeypatch.setattr(cli.SemanticServiceClient, "from_settings", factory)
    assert dataagent_main(["history-sql", "--if-enabled"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "disabled"
    factory.assert_not_called()
    assert not (tmp_path / "history_sql").exists()


def test_cli_downloads_native_config_and_explicit_output_wins(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "snapshot"
    rt = SimpleNamespace(
        settings=SimpleNamespace(
            plugins=SimpleNamespace(config={"de-agent": {"history_sql": {"enabled": True}}}),
            semantic=object(),
        ),
        paths=SimpleNamespace(home=tmp_path),
        redaction_secrets=(),
    )
    monkeypatch.setattr(cli, "prepare_runtime", lambda _: rt)
    monkeypatch.setenv("DATAAGENT_HISTORY_SQL_CORPUS", "relative-environment-path")
    with SemanticServiceClient(
        "http://semantic",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "entities": [] if int(req.url.params["offset"]) else [entity()],
                },
            )
        ),
    ) as client:
        monkeypatch.setattr(cli.SemanticServiceClient, "from_settings", lambda _: client)
        assert (
            dataagent_main(
                ["history-sql", "--if-enabled", "--output-dir", str(destination), "--no-lock"]
            )
            == 0
        )
    report = json.loads(capsys.readouterr().out)
    assert report["written"] == 1 and report["status"] == "ready"
    assert (destination / "DEMO.sql").read_text() == "SELECT 1\n"


def test_history_child_has_only_local_read_tools_and_private_skill():
    spec = SubAgentSpec.model_validate(read_yaml((PLUGIN / "subagents/history-sql-recall.yaml")))
    assert spec.tools == []
    assert set(spec.filesystem_tools) == {"ls", "glob", "grep", "read_file"}
    assert spec.skills == ["history-skills"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_enabled", [False, True])
async def test_native_task_reads_corpus_without_semantic_calls(tmp_path, is_enabled):
    root = tmp_path / "corpus"
    root.mkdir()
    sql = root / "DOWNLOAD.sql"
    sql.write_text("INSERT OVERWRITE db.demo SELECT user_id FROM db.download GROUP BY user_id")
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        json.dumps(
            {
                "models": {
                    "primary": {
                        "name": "test",
                        "base_url": "https://example.invalid/v1",
                        "api_key": "offline",
                    }
                },
                "plugins": {
                    "enabled": ["de-agent"],
                    "config": {
                        "de-agent": {
                            "history_sql": {"enabled": is_enabled, "corpus_path": str(root)},
                            "dataops": {"enabled": False},
                        }
                    },
                },
            }
        )
    )
    rt = prepare_runtime(LaunchOptions(cwd=tmp_path, config=cfg))
    child_responses = (
        [
            call(
                "read_file",
                {"file_path": str(PLUGIN / "history-skills/history-sql-retrieval/SKILL.md")},
                "skill",
            ),
            call("grep", {"pattern": "db.download", "path": str(root), "glob": "*.sql"}, "search"),
            call("read_file", {"file_path": str(sql)}, "file"),
            AIMessage(content="matched；来源 DOWNLOAD.sql；输出主键 user_id；仅供参考"),
        ]
        if is_enabled
        else []
    )
    model = ScriptedModel(
        responses=[
            call(
                "task",
                {
                    "subagent_type": "history-sql-recall",
                    "description": "下载次数；输出主键 user_id",
                },
                "history",
            ),
            *child_responses,
            AIMessage(content="检索完成"),
        ]
    )
    async with create_agent(rt, model=model) as agent:
        result = await agent.ainvoke({"messages": [HumanMessage(content="检索下载历史SQL")]})
    assert result["messages"][-1].content == "检索完成"
    assert len(model.requests) == (6 if is_enabled else 2)
    if is_enabled:
        assert "db.download" in str(model.requests[-2])
    else:
        assert '"status": "disabled"' in str(model.requests[-1])


def test_nl2sql_receives_reference_only_for_the_explicit_task(tmp_path):
    ctx = context(tmp_path)
    ctx.session.outputs.mkdir()
    (ctx.session.outputs / "create_temp.ads_demo_ds.sql").write_text(
        "CREATE EXTERNAL TABLE IF NOT EXISTS temp.ads_demo_ds "
        "(row_total BIGINT COMMENT '总行数') COMMENT '计数' "
        "PARTITIONED BY (pt_d STRING COMMENT '日期分区') STORED AS ORC;"
    )
    loader = PythonLoader()
    save = loader.load(PLUGIN, "utils/evidence.py:save_evidence")
    save(ctx, "schema", "db.download", {"db.download.user_id": {"value_type": "STRING"}})
    prepare = loader.load(PLUGIN, "hooks/nl2sql.py:prepare_input")
    args = {"question": "生成 temp.ads_demo_ds", "source_table": ["db.download"]}
    args["history_sql_reference"] = "来源 DOWNLOAD.sql：原文使用 COUNT(*)；仅作写法参考。"
    result = prepare({"question": json.dumps(args)}, SimpleNamespace(context=ctx), params={})
    evidence = json.loads(result["evidence"])
    assert evidence["history_sql_reference"]["trust"] == "unverified_reference"
    assert evidence["history_sql_reference"]["summary"] == args.pop("history_sql_reference")
    result = prepare({"question": json.dumps(args)}, SimpleNamespace(context=ctx), params={})
    assert "history_sql_reference" not in json.loads(result["evidence"])
    records = [json.loads(p.read_text()) for p in (ctx.session.outputs / "metadata").glob("*.json")]
    assert [record["kind"] for record in records] == ["schema"]


def _wrapper(payload):
    return {"type": "jsonb", "value": json.dumps(payload), "null": False}


def test_unwrap_jsonb_handles_pgobject_wrapper_and_passthrough_forms():
    assert _unwrap_jsonb(_wrapper({"is_verified": True})) == {"is_verified": True}
    plain = {"is_verified": True}
    assert _unwrap_jsonb(plain) is plain
    extra = {"type": "jsonb", "value": "x", "null": False, "extra": 1}
    assert _unwrap_jsonb(extra) is extra
    assert _unwrap_jsonb(json.dumps({"a": 1})) == {"a": 1}
    assert _unwrap_jsonb(json.dumps([1])) is None
    assert _unwrap_jsonb("not json") is None
    assert _unwrap_jsonb({"type": "jsonb", "value": "not json", "null": False}) is None
    assert _unwrap_jsonb({"type": "jsonb", "value": None, "null": True}) is None
    assert _unwrap_jsonb(None) is None
    assert _unwrap_jsonb(123) is None


def test_is_verified_sql_accepts_boolean_or_case_insensitive_true_string():
    def with_props(props):
        return entity("g", sql="SELECT 1", sql_process_properties=props)

    assert _is_verified_sql(with_props(_wrapper({"is_verified": True})))
    assert _is_verified_sql(with_props({"is_verified": True}))
    assert _is_verified_sql(with_props({"is_verified": " TRUE "}))
    assert not _is_verified_sql(with_props(_wrapper({"is_verified": False})))
    assert not _is_verified_sql(with_props({"is_verified": "false"}))
    assert not _is_verified_sql(with_props({"is_verified": "1"}))
    assert not _is_verified_sql(with_props({"other_key": True}))
    assert not _is_verified_sql(entity("g", sql="SELECT 1"))


def test_verified_only_write_keeps_verified_prunes_unverified(tmp_path):
    root = tmp_path / "corpus"
    rows = [
        entity(
            "keep",
            table="db.a",
            sql="SELECT 1",
            sql_process_properties=_wrapper({"is_verified": True}),
        ),
        entity(
            "drop",
            table="db.b",
            sql="SELECT 2",
            sql_process_properties=_wrapper({"is_verified": False}),
        ),
    ]
    stats = write_corpus(rows, root, lock=False)
    assert stats.written == 2
    manifest = json.loads((root / ".history_sql_manifest.json").read_text())
    assert manifest["verified_only"] is False

    stats = write_corpus(rows, root, lock=False, verified_only=True)
    assert stats.written == 1
    assert (root / "A.sql").is_file()
    assert not (root / "B.sql").exists()
    assert ("drop", "skip:unverified") in stats.skipped
    assert stats.pruned == ["B.sql"]
    manifest = json.loads((root / ".history_sql_manifest.json").read_text())
    assert manifest["verified_only"] is True


def test_cli_verified_only_flag_filters_the_snapshot(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "snapshot"
    rt = SimpleNamespace(
        settings=SimpleNamespace(
            plugins=SimpleNamespace(config={"de-agent": {"history_sql": {"enabled": True}}}),
            semantic=object(),
        ),
        paths=SimpleNamespace(home=tmp_path),
        redaction_secrets=(),
    )
    monkeypatch.setattr(cli, "prepare_runtime", lambda _: rt)

    def respond(req):
        rows = (
            [
                entity(
                    "keep",
                    table="db.a",
                    sql="SELECT 1",
                    sql_process_properties={"is_verified": True},
                ),
                entity(
                    "drop",
                    table="db.b",
                    sql="SELECT 2",
                    sql_process_properties={"is_verified": "false"},
                ),
            ]
            if int(req.url.params["offset"]) == 0
            else []
        )
        return httpx.Response(200, json={"entities": rows})

    with SemanticServiceClient("http://semantic", transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(cli.SemanticServiceClient, "from_settings", lambda _: client)
        assert (
            dataagent_main(
                [
                    "history-sql",
                    "--if-enabled",
                    "--output-dir",
                    str(destination),
                    "--no-lock",
                    "--verified-only",
                ]
            )
            == 0
        )
    report = json.loads(capsys.readouterr().out)
    assert report["written"] == 1
    assert report["skipped"] == [["drop", "skip:unverified"]]
    assert report["unrecognized_properties"] == 0
    assert (destination / "A.sql").is_file()
    assert not (destination / "B.sql").exists()


def test_verified_state_classifies_annotation_shapes():
    def with_props(props):
        return entity("g", sql="SELECT 1", sql_process_properties=props)

    assert _verified_state(with_props(_wrapper({"is_verified": True}))) == "verified"
    assert _verified_state(with_props({"is_verified": "TRUE"})) == "verified"
    assert _verified_state(with_props(_wrapper({"is_verified": False}))) == "unverified"
    assert _verified_state(with_props({"is_verified": "false"})) == "unverified"
    assert _verified_state(entity("g", sql="SELECT 1")) == "unverified"  # 列缺失
    # 包装形显式 NULL = 无标注（正常态），不算形态漂移
    assert (
        _verified_state(with_props({"type": "jsonb", "value": None, "null": True})) == "unverified"
    )
    # 有值但读不出 = 形态不可辨（疑似服务端字段漂移）
    assert (
        _verified_state(with_props({"type": "jsonb", "value": "not json", "null": False}))
        == "unrecognized"
    )
    assert _verified_state(with_props(json.dumps([1, 2]))) == "unrecognized"
    assert _verified_state(with_props(123)) == "unrecognized"
    # 服务端未来若把已解析对象直接装进包装 value，同样计为不可辨（可见而非静默）
    assert (
        _verified_state(
            with_props({"type": "jsonb", "value": {"is_verified": True}, "null": False})
        )
        == "unrecognized"
    )


def test_verified_only_counts_unreadable_shapes_without_changing_membership(tmp_path):
    rows = [
        entity(
            "keep",
            table="db.a",
            sql="SELECT 1",
            sql_process_properties=_wrapper({"is_verified": True}),
        ),
        entity(
            "explicit",
            table="db.b",
            sql="SELECT 2",
            sql_process_properties=_wrapper({"is_verified": False}),
        ),
        entity(
            "drift",
            table="db.c",
            sql="SELECT 3",
            sql_process_properties={"type": "jsonb", "value": "not json", "null": False},
        ),
    ]
    stats = write_corpus(rows, tmp_path / "corpus", lock=False, verified_only=True)
    assert stats.written == 1
    assert (tmp_path / "corpus" / "A.sql").is_file()
    assert not (tmp_path / "corpus" / "B.sql").exists()
    assert not (tmp_path / "corpus" / "C.sql").exists()
    assert stats.unrecognized_properties == 1  # 漂移条目计数；成员判定不变（照旧排除）
    assert ("explicit", "skip:unverified") in stats.skipped
    assert ("drift", "skip:unverified") in stats.skipped

    full = write_corpus(rows, tmp_path / "full", lock=False)  # 默认模式不感知标注形态
    assert full.written == 3
    assert full.unrecognized_properties == 0
