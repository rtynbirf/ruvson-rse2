#!/usr/bin/env python3
# Волна ×3 · W3-E · «DAG-манифест + золотые пробы» (ruvson-rse2#1): stages в steps.yaml,
# чекпоинты checkpoints/step_N.json {outs, sha256, версия модели, seed, длительность},
# resume с первого невалидного шага, гейт золотых проб goldens/step_N/manifest.json;
# stdlib, без JVM/conda/сторонних экшенов; схема парсера steps.yaml — без PyYAML.
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

CHECKDIR = "checkpoints"
NAMES = ["source", "separate", "vc-convert", "voice-loop", "mix", "master", "package"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def val(s):
    s = s.strip()
    if s in ('""', "''", ""): return ""
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [] if not inner else [x.strip() for x in inner.split(",")]
    return int(s) if re.fullmatch(r"-?\d+", s) else s

def load_steps(path):
    steps, cur = [], None
    for line in open(path, encoding="utf-8"):
        s = line.rstrip("\n")
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        m = re.match(r"^  - (\w+): (.*)$", s)
        if m:
            cur = {m.group(1): val(m.group(2))}; steps.append(cur); continue
        m = re.match(r"^    (\w+): (.*)$", s)
        if m and cur is not None:
            cur[m.group(1)] = val(m.group(2))
    for st in steps:
        st.setdefault("deps", []); st.setdefault("outs", []); st.setdefault("cmd", "")
        st.setdefault("model_version", ""); st.setdefault("seed", 42); st.setdefault("timeout_min", 30)
    return steps

def outs_state(step):
    st = {}
    for out in step["outs"]:
        p = Path(out)
        files = sorted(p.rglob("*")) if p.is_dir() else ([p] if p.is_file() else [])
        for f in files:
            if f.is_file():
                st[str(f)] = sha256(f)
    return st

def cp_path(sid): return Path(CHECKDIR) / ("%s.json" % sid)

def step_valid(step):
    if not cp_path(step["id"]).exists():
        return False
    cp = json.load(open(cp_path(step["id"]), encoding="utf-8"))
    return cp.get("outs") == outs_state(step) and cp.get("model_version") == step["model_version"]

def run_step(step):
    t0 = time.perf_counter()
    if not step["cmd"]:
        print("RUN %s — cmd не задан (шаблон): исполнение на станке оператора" % step["id"])
        return False
    env = dict(os.environ, RSE2_STEP=step["id"],
               RSE2_OUT=step["outs"][0] if step["outs"] else "", RSE2_OUTS=" ".join(step["outs"]))
    subprocess.run(step["cmd"], shell=True, check=True, env=env, timeout=step["timeout_min"] * 60)
    cp = {"step": step["id"], "model_version": step["model_version"], "seed": step["seed"],
          "duration_s": round(time.perf_counter() - t0, 3),
          "finished_at": datetime.now(timezone.utc).isoformat(), "outs": outs_state(step)}
    Path(CHECKDIR).mkdir(parents=True, exist_ok=True)
    with open(cp_path(step["id"]), "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    return True

def resume(steps):
    plan, broken = [], False
    for step in steps:
        if not broken and step_valid(step):
            plan.append(("SKIP", step["id"]))
        else:
            broken = True
            plan.append(("RUN", step["id"]))
    by_id = {s["id"]: s for s in steps}
    for act, sid in plan:
        if act == "SKIP":
            print("SKIP %s (sha256 чекпоинта валиден)" % sid)
        else:
            print("RUN %s" % sid)
            run_step(by_id[sid])
    return plan

def goldens(steps):  # → (найдены_ли, [(шаг, метрика, деталь)])
    fails, found = [], False
    for step in steps:
        g = Path("goldens") / step["id"] / "manifest.json"
        if not g.exists():
            continue
        found = True
        man = json.load(open(g, encoding="utf-8"))
        if not cp_path(step["id"]).exists():
            fails.append((step["id"], "checkpoint", "нет чекпоинта — не с чем сверить"))
            continue
        cp = json.load(open(cp_path(step["id"]), encoding="utf-8"))
        if man.get("model_version") != cp.get("model_version"):
            fails.append((step["id"], "model_version", "эталон %s, чекпоинт %s" % (man.get("model_version"), cp.get("model_version"))))
        for rel, h in sorted((man.get("outs") or {}).items()):
            cur = (cp.get("outs") or {}).get(rel)
            if cur != h:
                fails.append((step["id"], "sha256", "%s: эталон %s…, факт %s…" % (rel, (h or "—")[:12], (cur or "—")[:12])))
    return found, fails

def steps_text(cmd='""', model='""', timeout=60):
    lines = ["# steps.yaml · DAG-манифест RSE-2, шаги 0-6 (волна ×3, W3-E).",
             "# deps — входы, outs — выходы, cmd — команда, model_version — метка модели.", "steps:"]
    for n, name in enumerate(NAMES):
        dep = "[]" if n == 0 else "[artifacts/step_%d]" % (n - 1)
        lines += ["  - id: step_%d" % n, "    name: %s" % name, "    deps: %s" % dep,
                  "    outs: [artifacts/step_%d]" % n, "    cmd: %s" % cmd,
                  "    model_version: %s" % model, "    seed: 42", "    timeout_min: %d" % timeout]
    return "\n".join(lines) + "\n"

def init(path):
    if Path(path).exists():
        print("steps.yaml уже существует — DAG-манифест не перезаписывается (append-only).")
        return
    Path(path).write_text(steps_text() + "# cmd и model_version заполняет оператор на станке"
                                        " (заполняется при первом прогоне).\n", encoding="utf-8")
    print("steps.yaml создан: 7 ступеней DAG; заполните cmd/model_version на станке.")

def selftest():  # синтетический DAG 0-6: краш шага 4, подмена шага 2, подмена версии модели
    tmp = tempfile.mkdtemp(prefix="rse2-selftest-")
    old = os.getcwd()
    os.chdir(tmp)
    try:
        cmd = ("python3 -c \"import os,pathlib; d=pathlib.Path(os.environ['RSE2_OUT']); "
               "d.mkdir(parents=True, exist_ok=True); (d/'out.txt').write_text('art-' + os.environ['RSE2_STEP'])\"")
        Path("steps.yaml").write_text(steps_text(cmd=cmd, model="v1", timeout=5), encoding="utf-8")
        steps = load_steps("steps.yaml")
        assert [a for a, _ in resume(steps)] == ["RUN"] * 7  # эталонный первый прогон
        for sid in ("step_4", "step_5", "step_6"):  # имитация краша шага 4: чекпоинтов нет
            cp_path(sid).unlink()
        plan = resume(steps)  # критерий 1: голова из кэша, исполняется только хвост
        assert [s for a, s in plan if a == "SKIP"] == ["step_0", "step_1", "step_2", "step_3"] and \
            [a for a, _ in plan] == ["SKIP"] * 4 + ["RUN"] * 3, plan
        with open("artifacts/step_2/out.txt", "a", encoding="utf-8") as f:  # критерий 2: подмена
            f.write("TAMPER")
        plan = resume(steps)
        assert [s for a, s in plan if a == "SKIP"] == ["step_0", "step_1"] and \
            [a for a, _ in plan] == ["SKIP"] * 2 + ["RUN"] * 5, plan
        Path("goldens/step_3").mkdir(parents=True)  # критерий 3: золотая проба шага 3
        cp3 = json.load(open(cp_path("step_3"), encoding="utf-8"))
        json.dump({"model_version": cp3["model_version"], "outs": cp3["outs"]},
                  open("goldens/step_3/manifest.json", "w", encoding="utf-8"))
        found, fails = goldens(steps)
        assert found and not fails, fails
        cp3["model_version"] = "roformer-vX"  # подмена метки версии модели в чекпоинте шага 3
        json.dump(cp3, open(cp_path("step_3"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        found, fails = goldens(steps)
        assert found and fails and fails[0][0] == "step_3" and fails[0][1] == "model_version", fails
        print("SELFTEST OK: SKIP по sha256 работает; DAG пересчитывает ровно шаги 2-6; "
              "гейт краснеет: шаг %s, метрика %s (%s)" % fails[0])
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser(description="RSE-2 DAG: чекпоинты, resume, золотые пробы")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "resume", "goldens", "selftest"):
        sub.add_parser(name)
    a = ap.parse_args()
    if a.cmd == "init":
        init("steps.yaml")
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "resume":
        plan = resume(load_steps("steps.yaml"))
        ran = sum(1 for act, _ in plan if act == "RUN")
        print("RESUME: выполнено %d, из кэша %d" % (ran, len(plan) - ran))
    elif a.cmd == "goldens":
        if not Path("steps.yaml").exists():
            print("steps.yaml ещё не создан (rse2.py init) — гейт золотых проб в режиме ожидания.")
        elif not any((Path("goldens") / s["id"]).exists() for s in load_steps("steps.yaml")):
            print("Золотые пробы не заполнены (goldens/step_N/manifest.json) — гейт в режиме ожидания.")
        else:
            found, fails = goldens(load_steps("steps.yaml"))
            for sid, metric, detail in fails:
                print("ГЕЙТ ЗОЛОТЫХ ПРОБ: КРАСНЫЙ — шаг %s, метрика %s: %s" % (sid, metric, detail))
            if fails:
                sys.exit(1)
            print("ГЕЙТ ЗОЛОТЫХ ПРОБ: ЗЕЛЁНЫЙ — дрейфа нет.")

if __name__ == "__main__":
    main()
