# This file is generated by gyp; do not edit.

TOOLSET := target
TARGET := addon
DEFS_Debug := \
	'-D_LARGEFILE_SOURCE' \
	'-D_FILE_OFFSET_BITS=64' \
	'-DBUILDING_NODE_EXTENSION' \
	'-DDEBUG' \
	'-D_DEBUG'

# Flags passed to all source files.
CFLAGS_Debug := \
	-fPIC \
	-Wall \
	-Wextra \
	-Wno-unused-parameter \
	-pthread \
	-m64 \
	-g \
	-O0 \
	$(RPM_OPT_FLAGS)

# Flags passed to only C files.
CFLAGS_C_Debug := \
	$(RPM_OPT_FLAGS)

# Flags passed to only C++ files.
CFLAGS_CC_Debug := \
	-fno-rtti \
	-fno-exceptions \
	$(RPM_OPT_FLAGS)

INCS_Debug := \
	-I/usr/include \
	-I/usr/include/node \
	-I$(srcdir)/node_modules/nan

DEFS_Release := \
	'-D_LARGEFILE_SOURCE' \
	'-D_FILE_OFFSET_BITS=64' \
	'-DBUILDING_NODE_EXTENSION'

# Flags passed to all source files.
CFLAGS_Release := \
	-fPIC \
	-Wall \
	-Wextra \
	-Wno-unused-parameter \
	-pthread \
	-m64 \
	-O2 \
	-fno-strict-aliasing \
	-fno-tree-vrp \
	-fno-omit-frame-pointer \
	$(RPM_OPT_FLAGS)

# Flags passed to only C files.
CFLAGS_C_Release := \
	$(RPM_OPT_FLAGS)

# Flags passed to only C++ files.
CFLAGS_CC_Release := \
	-fno-rtti \
	-fno-exceptions \
	$(RPM_OPT_FLAGS)

INCS_Release := \
	-I/usr/include \
	-I/usr/include/node \
	-I$(srcdir)/node_modules/nan

OBJS := \
	$(obj).target/$(TARGET)/AnnAddon.o \
	$(obj).target/$(TARGET)/AnnWrapper.o

# Add to the list of files we specially track dependencies for.
all_deps += $(OBJS)

# CFLAGS et al overrides must be target-local.
# See "Target-specific Variable Values" in the GNU Make manual.
$(OBJS): TOOLSET := $(TOOLSET)
$(OBJS): GYP_CFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE))  $(CFLAGS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE))
$(OBJS): GYP_CXXFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE))  $(CFLAGS_$(BUILDTYPE)) $(CFLAGS_CC_$(BUILDTYPE))

# Suffix rules, putting all outputs into $(obj).

$(obj).$(TOOLSET)/$(TARGET)/%.o: $(srcdir)/%.cpp FORCE_DO_CMD
	@$(call do_cmd,cxx,1)

# Try building from generated source, too.

$(obj).$(TOOLSET)/$(TARGET)/%.o: $(obj).$(TOOLSET)/%.cpp FORCE_DO_CMD
	@$(call do_cmd,cxx,1)

$(obj).$(TOOLSET)/$(TARGET)/%.o: $(obj)/%.cpp FORCE_DO_CMD
	@$(call do_cmd,cxx,1)

# End of this set of suffix rules
### Rules for final target.
LDFLAGS_Debug := \
	-pthread \
	-rdynamic \
	-m64

LDFLAGS_Release := \
	-pthread \
	-rdynamic \
	-m64

LIBS :=

$(obj).target/addon.node: GYP_LDFLAGS := $(LDFLAGS_$(BUILDTYPE))
$(obj).target/addon.node: LIBS := $(LIBS)
$(obj).target/addon.node: TOOLSET := $(TOOLSET)
$(obj).target/addon.node: $(OBJS) FORCE_DO_CMD
	$(call do_cmd,solink_module)

all_deps += $(obj).target/addon.node
# Add target alias
.PHONY: addon
addon: $(builddir)/addon.node

# Copy this to the executable output path.
$(builddir)/addon.node: TOOLSET := $(TOOLSET)
$(builddir)/addon.node: $(obj).target/addon.node FORCE_DO_CMD
	$(call do_cmd,copy)

all_deps += $(builddir)/addon.node
# Short alias for building this executable.
.PHONY: addon.node
addon.node: $(obj).target/addon.node $(builddir)/addon.node

# Add executable to "all" target.
.PHONY: all
all: $(builddir)/addon.node

