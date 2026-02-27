# Jittor macOS ARM64 编译修复

Jittor v1.3.10 在 macOS ARM64 (Apple Silicon) 上无法直接编译运行，需要对底层代码做 3 处修改（共约 31 行改动）。

本目录包含修改后的完整文件，原始文件位于 `site-packages/jittor/` 下。

## 使用方法

```bash
# 找到 jittor 安装路径
JITTOR_DIR=$(python3 -c "import jittor; import os; print(os.path.dirname(jittor.__file__))")

# 备份并替换
cp "$JITTOR_DIR/compiler.py" "$JITTOR_DIR/compiler.py.bak"
cp "$JITTOR_DIR/src/opt/expr.h" "$JITTOR_DIR/src/opt/expr.h.bak"
cp "$JITTOR_DIR/src/misc/stack_vector.h" "$JITTOR_DIR/src/misc/stack_vector.h.bak"

cp compiler.py "$JITTOR_DIR/compiler.py"
cp expr.h "$JITTOR_DIR/src/opt/expr.h"
cp stack_vector.h "$JITTOR_DIR/src/misc/stack_vector.h"
```

## 修复说明

### ❶ compiler.py — PMF 指针强转修复（+27 行）

**位置**: 第 1363–1389 行

macOS clang 严格遵循 C++ 标准，禁止成员函数指针 (PMF) → 普通函数指针的强制转换：

```cpp
// ❌ 原始写法（Linux GCC 可通过，macOS clang 报错）
(void(*)(Node*)) &Node::forward

// ✅ 修复后（基于 union 类型双关的安全转换）
unsafe_pmf_cast<void(*)(Node*)>(&Node::forward)
```

修复方案：
1. `clang -E` 预处理源码 → 展开后的 `.pp.cc` 文件
2. 注入 `unsafe_pmf_cast` 模板函数（union 类型双关）
3. 正则替换所有 PMF 强转
4. 编译修改后的文件，清理临时文件
5. 仅 `platform.system() == 'Darwin'` 时触发

### ❷ expr.h — std::move 限定修复（2 处改动）

**位置**: 第 160、164 行

macOS clang 要求 `move()` 必须带 `std::` 前缀：

```cpp
// ❌ 原始
children.emplace_back(move(c));
return make(str, move(children));

// ✅ 修复
children.emplace_back(std::move(c));
return make(str, std::move(children));
```

### ❸ stack_vector.h — 补全 at() 方法（+2 行）

**位置**: 第 35–36 行

macOS 标准库某些代码路径调用 `at()` 而非 `operator[]`：

```cpp
// 新增
inline const T& at(int i) const { return a[i]; }
inline T& at(int i) { return a[i]; }
```
