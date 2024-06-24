---
{"dg-publish":true,"permalink":"/tasks/","created":"2024-05-27T14:56:31.780+08:00"}
---

### Not done

```tasks
[not done] AND [path includes Plans/Daily Notes] AND [path does not include Plans/Daily Notes/archived] AND [description regex does not match /[012].*/]
limit 10
sort by due
```

### Daily

```tasks
[not done] AND [path includes Plans/Daily Notes] AND [path does not include Plans/Daily Notes/archived] AND [description regex matches /[012].*/]
limit 10
sort by due
```
    