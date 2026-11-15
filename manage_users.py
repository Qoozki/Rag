#!/usr/bin/env python
"""Manage users:  python manage_users.py add alice --role admin | passwd | remove | list"""
import argparse
import getpass
import sys

import database
import config


def _read_password(args) -> str:
    if args.password:           # discouraged (shell history) but handy for scripts
        return args.password
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    return first


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("add", "passwd"):
        sp = sub.add_parser(name)
        sp.add_argument("username")
        sp.add_argument("--password")
        if name == "add":
            sp.add_argument("--role", choices=sorted(database.ROLES), default="viewer")
    sub.add_parser("remove").add_argument("username")
    sub.add_parser("list")
    args = p.parse_args(argv)

    users = database.load_users_file(config.USERS_FILE) if __import__("os").path.exists(config.USERS_FILE) else {}

    if args.cmd == "list":
        for u in users.values():
            print(f"{u.username}\t{u.role}")
        return 0
    if args.cmd == "remove":
        if users.pop(args.username, None) is None:
            sys.exit(f"No such user: {args.username}")
    else:
        if args.cmd == "add" and args.username in users:
            sys.exit("User already exists (use `passwd` to change the password).")
        if args.cmd == "passwd" and args.username not in users:
            sys.exit(f"No such user: {args.username}")
        password = _read_password(args)
        if len(password) < config.MIN_PASSWORD_LENGTH:
            sys.exit(f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters.")
        role = args.role if args.cmd == "add" else users[args.username].role
        users[args.username] = database.User(args.username, role, database.hash_password(password))
    database.save_users_file(config.USERS_FILE, users)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
