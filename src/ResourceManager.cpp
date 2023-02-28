#include "ResourceManager.h"
#include <errno.h>
#include <exception>

/// <summary>
/// Lua error class
/// </summary>
class InvalidTypeError : public std::exception
{
public:
	InvalidTypeError(const char* msg) : std::exception(msg) {}
};

////////////////////////////////////////////////////////////
bool CheckLua(lua_State* L, int r)
{
	if (r != LUA_OK)
	{
		std::string errormsg = lua_tostring(L, -1);
		std::cout << errormsg << std::endl;
		return false;
	}
	return true;
}

////////////////////////////////////////////////////////////
ResourcesManager::ResourcesManager(std::string filename)
{
	m_L = luaL_newstate();
	luaL_openlibs(m_L);
	if (!CheckLua(m_L, luaL_dofile(m_L, filename.c_str())))
	{
		throw std::ios_base::failure("Configuration file: " + filename + " doesnt exists");
	}
}

////////////////////////////////////////////////////////////
ResourcesManager::~ResourcesManager()
{
	lua_close(m_L);
}

////////////////////////////////////////////////////////////
std::string ResourcesManager::getString(std::string name)
{
	lua_getglobal(m_L, name.c_str());
	if (lua_isstring(m_L, -1))
	{
		return lua_tostring(m_L, -1);
	}
	else
	{
		throw InvalidTypeError("Required type is string");
	}
}

////////////////////////////////////////////////////////////
int ResourcesManager::getInt(std::string name)
{
	lua_getglobal(m_L, name.c_str());
	if (lua_isnumber(m_L, -1))
	{
		return (int)lua_tointeger(m_L, -1);
	}
	else
	{
		throw InvalidTypeError("Required type is int");
	}
}

////////////////////////////////////////////////////////////
float ResourcesManager::getFloat(std::string name)
{
	lua_getglobal(m_L, name.c_str());
	if (lua_isnumber(m_L, -1))
	{
		return (float)lua_tonumber(m_L, -1);
	}
	else
	{
		throw InvalidTypeError("Required type is float");
	}
}