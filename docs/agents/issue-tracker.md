# Issue tracker: GitHub

任务与规格保存在 [ConutlessLoneSmokes/DMA_CE Issues](https://github.com/ConutlessLoneSmokes/DMA_CE/issues)。同一项工作更新已有 issue，相关实验用运行 ID 和仓库路径连接证据。

优先使用已连接的 GitHub 工具读取、创建和更新 issue；需要 CLI 时在 `gh` 对应子命令中指定 `--repo ConutlessLoneSmokes/DMA_CE`。多行正文写入 UTF-8 文件，再用 `--body-file`，或直接传给工具的结构化正文参数。当前本机未安装 `gh`，已有 GitHub 连接可完成 issue 操作；缺失的接口可用现有 Git 凭据调用 GitHub API，凭据仅在进程内使用。

## 常用操作

- 读取：`gh issue view <number> --repo ConutlessLoneSmokes/DMA_CE --json number,title,body,labels,comments`。
- 列表：`gh issue list --repo ConutlessLoneSmokes/DMA_CE --state open --json number,title,labels`，按需限定标签。
- 创建：`gh issue create --repo ConutlessLoneSmokes/DMA_CE --title "标题" --body-file <正文文件>`。
- 更新：`gh issue edit <number> --repo ConutlessLoneSmokes/DMA_CE --body-file <正文文件>`；标签使用 `--add-label` / `--remove-label`。
- 评论：`gh issue comment <number> --repo ConutlessLoneSmokes/DMA_CE --body-file <正文文件>`；验收满足后关闭 issue。

skill 中的 “publish to the issue tracker” 指创建或更新 GitHub issue；“fetch the relevant ticket” 指读取对应 issue 及相关评论。

**PRs as a request surface: no.**

## 分解与关联

父任务保存问题、范围、验收条件和子任务链接；子任务使用 GitHub 原生子 issue 与依赖关系。工具不支持时，在子任务开头记录 `Part of #<parent>`、`Blocked by: #<number>`，并在父任务中保留任务清单。

调用 `wayfinder` 时以一个 `wayfinder:map` issue 保存 Notes / Decisions-so-far / Fog；子任务按 `wayfinder:<type>` 区分 research、prototype、grilling、task。有未关闭依赖的任务等待依赖；领取后分配给执行者，完成后将答案与证据链接写回父任务。这些额外标签仅在实际采用该流程时建立。

执行范围以 issue、会话授权和 [工作流程](../WORKFLOW.md) 为准；任务标签本身不扩展实验预算。
