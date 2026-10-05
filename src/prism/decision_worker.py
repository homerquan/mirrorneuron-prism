"""Spawn target: ML imports and resident weights stay outside the ASGI process."""


def serve(connection, config):
    try:
        import laya

        agent = laya.load(
            config["model"],
            device="cpu",
            revision=config["revision"],
            expected_sha256=config["expected_sha256"],
        )
        connection.send({"ready": True})
        while True:
            job = connection.recv()
            if job is None:
                break
            try:
                result = agent.system_one(
                    job["state"],
                    job["questions"],
                    max_len=config["max_len"],
                    head_max_len=256,
                )
                connection.send({"result": result})
            except Exception:
                connection.send({"error": "decision_unavailable"})
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            connection.send({"error": "preparation_failed"})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()
