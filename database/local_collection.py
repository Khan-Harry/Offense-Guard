"""
LocalCollection & LocalCursor
Serverless fallback storage for local/development environments or when MongoDB is unreachable.
"""

import os
import json
import secrets
import tempfile
from datetime import datetime


class LocalCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key, direction=1):
        try:
            self._docs.sort(key=lambda x: str(x.get(key, '')), reverse=(direction == -1))
        except Exception:
            pass
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)

    def __getitem__(self, index):
        return self._docs[index]


class LocalCollection:
    def __init__(self, name, initial_docs=None):
        self.name = name
        self.filename = os.path.join(tempfile.gettempdir(), f"og_{name}.json")
        self.docs = []
        self._load()
        if not self.docs and initial_docs:
            self.docs = list(initial_docs)
            self._save()

    def _load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.docs = json.load(f)
            except Exception:
                pass

    def _save(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.docs, f, default=str)
        except Exception:
            pass

    def _matches(self, doc, query):
        if not query:
            return True
        for k, v in query.items():
            if k == '$or':
                if not any(self._matches(doc, q) for q in v):
                    return False
            elif doc.get(k) != v:
                return False
        return True

    def find_one(self, query=None):
        self._load()
        for doc in self.docs:
            if self._matches(doc, query or {}):
                return doc.copy()
        return None

    def find(self, query=None, projection=None):
        self._load()
        res = []
        for doc in self.docs:
            if self._matches(doc, query or {}):
                d = doc.copy()
                if projection:
                    for pk, pv in projection.items():
                        if pv == 0 and pk in d:
                            del d[pk]
                res.append(d)
        return LocalCursor(res)

    def insert_one(self, doc):
        d = doc.copy()
        if '_id' not in d:
            d['_id'] = secrets.token_hex(12)
        for k, v in list(d.items()):
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        self.docs.append(d)
        self._save()
        return d

    def count_documents(self, query=None):
        self._load()
        return sum(1 for doc in self.docs if self._matches(doc, query or {}))

    def update_one(self, query, update, upsert=False):
        self._load()
        target = None
        for doc in self.docs:
            if self._matches(doc, query):
                target = doc
                break
        if target is None:
            if upsert:
                target = {'_id': secrets.token_hex(12)}
                for k, v in query.items():
                    if not k.startswith('$'):
                        target[k] = v
                self.docs.append(target)
            else:
                return None
        if '$set' in update:
            for k, v in update['$set'].items():
                if isinstance(v, datetime):
                    v = v.isoformat()
                target[k] = v
        if '$inc' in update:
            for k, v in update['$inc'].items():
                target[k] = target.get(k, 0) + v
        self._save()
        return target

    def delete_one(self, query):
        self._load()
        for i, doc in enumerate(self.docs):
            if self._matches(doc, query):
                del self.docs[i]
                self._save()
                return True
        return False

    def delete_many(self, query):
        self._load()
        orig_len = len(self.docs)
        self.docs = [d for d in self.docs if not self._matches(d, query)]
        if len(self.docs) != orig_len:
            self._save()
        return True
